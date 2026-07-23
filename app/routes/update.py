import datetime
import logging
import shutil
import os
import csv
from threading import Thread
from flask import Blueprint, current_app, request
from app.models import (
    ModelParameters,
    StudyData,
    ModelUpdateRequests,
    Group,
    Action,
    DataUpload,
    ThompsonSamplingParams,
    UpdateReproducibilitySnapshot,
)
from app.algorithms.base import RLAlgorithm
from app.extensions import db
from app.reward_derivation import derive_study_data
from app.repro_snapshot import save_pre_update_repro_snapshot
from app.routes.envelope import envelope, new_rid

update_blueprint = Blueprint("update", __name__)


def backup_tables(app):
    """
    Backs up all database tables into CSV files in a timestamped directory and zips them.
    """
    backup_dir = os.path.join(
        "backups", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    os.makedirs(backup_dir, exist_ok=True)

    with app.app_context():
        # List all models to back up
        models = [
            Group,
            DataUpload,
            Action,
            StudyData,
            ModelUpdateRequests,
            ModelParameters,
            ThompsonSamplingParams,
            UpdateReproducibilitySnapshot,
        ]

        for model in models:
            table_name = model.__tablename__
            file_path = os.path.join(backup_dir, f"{table_name}.csv")

            # Query all data from the table
            rows = db.session.query(model).all()

            # Extract column names
            columns = [col.key for col in model.__table__.columns]

            # Write the data to a CSV file
            with open(file_path, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(columns)
                for row in rows:
                    writer.writerow([getattr(row, col) for col in columns])

    # Zip the backup directory
    shutil.make_archive(backup_dir, "zip", backup_dir)

    # Clean up the backup directory
    shutil.rmtree(backup_dir)

    return f"{backup_dir}.zip"


def process_update_request(app, rid: str, rl_algorithm: RLAlgorithm):
    """
    Process the update request (API-Spec §2.4).

    No callback: completion is observed by reading model_update_requests.status
    / completed_at. Rewards are derived server-side from the data_uploads
    timeline before fitting. ``rid`` keys the model_update_requests row.
    """
    try:
        # Check if the database backup is enabled
        if app.config.get("BACKUP_DATABASE"):
            backup_file = backup_tables(app)
            app.logger.info("Database backed up to: %s", backup_file)

        with app.app_context():
            # Derive (action, outcome) pairs from the data_uploads timeline,
            # writing/refreshing study_data rows before the learner runs.
            n_derived = derive_study_data(app)
            app.logger.info("[Update] Derived %d study_data rows", n_derived)

            # Get the latest model parameters from the database
            current_params = ModelParameters.query.order_by(
                ModelParameters.timestamp.desc()
            ).first()

            # Get the data required for the update
            study_data = StudyData.query.order_by(
                StudyData.decision_type.asc(),
                StudyData.group_id.asc(),
                StudyData.decision_idx.asc(),
            ).all()

            records = []
            current_index = {}
            for row in study_data:
                agent_idx = int(row.raw_context.get("agent_decision_index", row.decision_idx + 1))
                records.append(
                    {
                        "group_id": row.group_id,
                        "decision_idx": row.decision_idx,
                        "decision_type": row.decision_type,
                        "agent_decision_index": agent_idx,
                        "state": row.state,
                        "action": row.action,
                        "reward": row.reward,
                        "raw_context": row.raw_context,
                        "outcome": row.outcome,
                    }
                )
                current_index[row.decision_type] = max(
                    current_index.get(row.decision_type, 0), agent_idx
                )

            update_data = {
                "records": records,
                "current_index": current_index,
            }

            snap_dir = save_pre_update_repro_snapshot(
                app, rid, current_params.id if current_params else None
            )
            if snap_dir:
                app.logger.info("Pre-update reproducibility snapshot: %s", snap_dir)

            status, new_parameters = rl_algorithm.update(
                {"probability_of_action": current_params.probability_of_action},
                update_data,
            )

            if not status:
                raise Exception("Model update failed.")

            # Add the new model parameters to the database
            new_model_parameters = ModelParameters(
                new_parameters["probability_of_action"]
            )

            db.session.add(new_model_parameters)
            db.session.commit()

            # Update the status of the request
            model_update_request = ModelUpdateRequests.query.filter_by(
                rid=rid
            ).first()
            model_update_request.status = "completed"
            model_update_request.completed_at = datetime.datetime.now()
            db.session.commit()

            # Log the completion
            logging.info(f"[Update] rid: {rid} completed.")

    except Exception as e:
        with app.app_context():
            # Log the error
            logging.error(f"[Update] Error: {e}")
            logging.exception(e)

            # Update the status of the request
            model_update_request = ModelUpdateRequests.query.filter_by(
                rid=rid
            ).first()
            if model_update_request is not None:
                model_update_request.status = "failed"
                model_update_request.completed_at = datetime.datetime.now()
                model_update_request.error_message = str(e)
                db.session.commit()

            # Log the completion
            logging.info(f"[Update] rid: {rid} failed.")


def check_fields(data: dict) -> tuple[bool, str]:
    """
    Check if the required fields are present in the data.
    """
    if not data or "timestamp" not in data:
        return False, "timestamp is required."

    return True, ""


@update_blueprint.route("/update", methods=["POST"])
def update_model():
    """
    Updates the algorithm model (API-Spec §2.4). Asynchronous; the monitoring
    algorithm triggers this and watches model_update_requests for completion.
    There is no callback. The always-present `rid` keys the
    model_update_requests row the monitoring algorithm polls.
    """
    rid = new_rid()
    try:
        data = request.get_json(silent=True)

        # Check if the required fields are present
        fields_present, error_message = check_fields(data)
        if not fields_present:
            return envelope(400, "Invalid Parameter", error_message, rid)

        # Extract the data
        request_timestamp = data["timestamp"]
        if isinstance(request_timestamp, str):
            request_timestamp = datetime.datetime.fromisoformat(request_timestamp)

        # Get the RL algorithm
        rl_algorithm = current_app.rl_algorithm
        logging.info(f"[Update] rid: {rid}")

        # Enqueue the update request. `rid` is the key the monitoring algorithm
        # polls; on an enqueue failure no model_update_requests row is created.
        try:
            model_update_request = ModelUpdateRequests(rid, request_timestamp)
            db.session.add(model_update_request)
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            logging.exception(exc)
            return envelope(
                503, "Service Unavailable",
                "Could not start the update; retry.", rid,
            )

        app = current_app._get_current_object()  # Get the actual app object
        if app.config.get("TESTING"):
            # Run inline under tests: a background thread sharing the in-memory
            # SQLite connection races the request thread's transaction.
            process_update_request(app, rid, rl_algorithm)
        else:
            # Process the update request in a separate thread.
            thread = Thread(
                target=process_update_request,
                args=(app, rid, rl_algorithm),
            )
            thread.start()

        return envelope(202, "Update Accepted", "Model update started.", rid)

    except Exception as e:
        # Log the error
        logging.error(f"[Update] Error: {e}")
        logging.exception(e)
        return envelope(503, "Service Unavailable", "Could not start the update; retry.", rid)
