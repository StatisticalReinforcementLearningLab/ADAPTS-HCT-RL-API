# ADAPTS-HCT

## Overview
ADAPTS-HCT is a digital intervention designed to improve medication adherence in adolescents and young adults (AYAs) who have undergone hematopoietic cell transplantation (HCT). After discharge, AYAs and their care partners face individual challenges (e.g., physical and emotional symptoms) and interpersonal barriers (e.g., family conflict) that can undermine adherence to critical immunosuppressant regimens. ADAPTS-HCT targets both members of the dyad and their relationship through three components: twice-daily positive psychology messages for the AYA, daily coping and self-care messages for the care partner, and a weekly collaborative game to strengthen the dyadic relationship.

## Interaction Flow

Here is a timeline of how the server will call this RL API to provide the RL algorithm with the context and request an action.

Every dyad when recruited will be active for 100 days. The server will call the RL API group.py to register the dyad.

Each week:
1. Sunday evening at 3AM EST, the server will call the RL API to update the model.
2. Sunday morning at 6AM EST, the server will call the RL API for every each active dyad (group in this repository) to request a **game action**.
3. Sunday morning at 9AM EST, the server will call the RL API for every each active AYA to request an AYA **message action**.
4. Sunday morning at 9AM EST, the server will call the RL API for every each active care partner to request a care partner **message action**.

## Simulation Logic

The study will recruit 25 dyads and each dyad will be active for 100 days. Your job is to simulate the server calls and data for this study. Note that the dyad recruitment is not sequential, i.e., we don't wait for the previous dyad to be completed before recruiting the next dyad. The recruitment rate is around 1 dyad per week.