from __future__ import annotations

import numpy as np
import warnings

from .action_space import action_index_to_duration, feasible_action_mask, feasible_actions, repair_action, get_action_set
from .bus_process import apply_charge, apply_travel_consumption
from .dwell_time import compute_dwell_breakdown
from .event_calendar import EventCalendar
from .parcel_process import compute_unloading_volume, get_unloading_parcels, unload_parcels_to_locker
from src.offline.assignment_io import build_assignment_indices
from .passenger_process import sample_poisson_arrivals, sample_initial_passenger_event, finalize_passenger_service
from .reward import compute_reward
from src.low_level.station_operator import operate_station_step
from .termination import apply_terminal_penalty_once, check_termination
from .state_builder import build_observation, build_feature_names


class EBusDroneEnv:
    def _bus_for_trip(self, trip_id: int) -> dict:
        return self.physical_bus_states[self.vehicle_circulation[int(trip_id)]]
    def _apply_non_service_return(self, bus: dict) -> None:
        op = self.instance.get("network", {})
        return_time = float(op.get("return_time_min", self.config.get("operation", {}).get("return_time_min", 0.0)))
        line_time = max(1e-9, float(op.get("nominal_line_time_min", self.config.get("operation", {}).get("nominal_line_time_min", return_time))))
        return_distance = float(self.state.get("travel_distance_km", 1.0)) * max(0.0, return_time / line_time)
        bus["battery_kwh"] = apply_travel_consumption(bus["battery_kwh"], return_distance, self.state["travel_energy_kwh_per_km"])
    def __init__(self, config: dict | None = None, instance: dict | None = None, scenario: dict | None = None, assignment: dict | None = None, smoke_test: bool = False):
        self.config = config or {}
        self.instance = instance
        self.scenario = scenario
        self.assignment = assignment
        self.smoke_test = smoke_test
        if not (self.instance and self.scenario and self.assignment) and not smoke_test:
            raise ValueError("EBusDroneEnv requires generated instance/scenario and offline assignment. Run generate then offline modes first.")
        self.terminal_penalty_applied = False
        self.reset()

    def reset(self, seed=None, options=None):
        _ = options
        if seed is not None:
            self.config["seed"] = int(seed)
        self._reset_from_data() if (self.instance and self.scenario and self.assignment) else self._reset_smoke()
        self.current_decision_event = None
        self.current_event = None
        self._advance_until_decision()
        self.terminal_penalty_applied = False
        return self._build_obs_for_current_event(), {"event": self.current_decision_event}

    def _reset_smoke(self):
        self.instance = {"horizon_minutes": 300.0, "network": {"stops": [{"stop_id": 0}, {"stop_id": 1}], "nominal_travel_time_min": [[0, 5], [5, 0]], "scheduled_bus_trips": [{"trip_id": 0, "departure_min": 0}]}, "stations": {"stations": [{"station_id": 1, "stop_id": 1, "chargers": 1, "drones": 1, "locker_capacity_kg": 100.0, "initial_fully_charged_batteries": 1, "initial_depleted_batteries": 0, "station_power_capacity_kw": 200.0}]}, "bus": {"battery_capacity_kwh": 150.0, "energy_kwh_per_km": 1.6}, "charging": {"pantograph_power_kw": 500.0, "charger_efficiency": 0.95}}
        self.scenario = {"passenger": {"passenger_arrivals": {"1": [0]}}}
        self.assignment = {"decisions": []}
        self._reset_from_data()

    def _reset_from_data(self):
        legacy_h = float(self.instance.get("horizon_minutes", 300.0))
        self.bus_operation_horizon = float(self.instance.get("bus_operation_horizon_minutes", legacy_h))
        self.delivery_evaluation_horizon = float(self.instance.get("delivery_evaluation_horizon_minutes", legacy_h))
        if self.delivery_evaluation_horizon < self.bus_operation_horizon:
            self.delivery_evaluation_horizon = self.bus_operation_horizon
        self.horizon = self.delivery_evaluation_horizon
        self.stop_ids = [int(s["stop_id"]) for s in self.instance["network"]["stops"]]
        self.trip_ids = [int(t["trip_id"]) for t in self.instance["network"]["scheduled_bus_trips"]]
        if "physical_buses" not in self.instance["network"] or "vehicle_circulation" not in self.instance["network"]:
            warnings.warn("Instance missing physical_buses or vehicle_circulation; falling back to trip-as-vehicle compatibility mode. Do not use for paper experiments.", RuntimeWarning)
            self.instance["network"]["physical_buses"] = [f"trip_{tid}" for tid in self.trip_ids]
            self.instance["network"]["vehicle_circulation"] = {int(tid): f"trip_{tid}" for tid in self.trip_ids}
        self.physical_buses = list(self.instance["network"]["physical_buses"])
        self.vehicle_circulation = {int(k): str(v) for k, v in self.instance["network"]["vehicle_circulation"].items()}
        self.freight_carrying_trip_ids = set(int(tid) for tid in self.instance["network"].get("freight_carrying_trip_ids", self.trip_ids))
        self.station_ids = [int(s["station_id"]) for s in self.instance["stations"]["stations"]]
        self.action_set = get_action_set(self.config)
        self.station_by_stop = {int(s.get("stop_id", s["station_id"])): int(s["station_id"]) for s in self.instance["stations"]["stations"]}
        self.travel_times = self.instance["network"]["nominal_travel_time_min"][0]
        self.assignment_index = build_assignment_indices(self.assignment)
        self.calendar = EventCalendar()
        self.event_log = []
        self.episode_metrics = {"number_decision_events": 0, "terminal_penalty": 0.0, "reward_component_sums": {}, "invalid_action_count": 0, "action_repair_count": 0, "requested_action_sum": 0.0, "executed_action_sum": 0.0, "action_gap_sum": 0.0, "requested_duration_min_sum": 0.0, "executed_duration_min_sum": 0.0, "positive_executed_duration_min_sum": 0.0, "charging_event_count": 0.0, "zero_charging_event_count": 0.0}
        self.rng = np.random.default_rng(int(self.config.get("seed", 0)))
        self.stop_queues = {sid: 0 for sid in self.stop_ids}
        self.stop_last_update = {sid: 0.0 for sid in self.stop_ids}

        cap = float(self.instance["bus"]["battery_capacity_kwh"])
        init_batt_min_frac = float(self.config.get("bus", {}).get("initial_battery_fraction_min", 1.0))
        init_batt_max_frac = float(self.config.get("bus", {}).get("initial_battery_fraction_max", 1.0))
        init_batt_min_frac = max(0.0, min(1.0, init_batt_min_frac))
        init_batt_max_frac = max(init_batt_min_frac, min(1.0, init_batt_max_frac))
        self.physical_bus_states = {}
        for vid in self.physical_buses:
            init_batt_kwh = float(self.rng.uniform(init_batt_min_frac * cap, init_batt_max_frac * cap))
            self.physical_bus_states[str(vid)] = {"vehicle_id": str(vid), "trip_id": -1, "current_stop_index": -1, "next_stop_id": self.stop_ids[0], "next_arrival_time_min": 0.0, "battery_kwh": init_batt_kwh, "battery_capacity_kwh": cap, "safety_battery_kwh": float(self.config.get("bus", {}).get("safety_battery_kwh", 5.0)), "onboard_passengers": 0, "passenger_capacity": int(self.config.get("bus", {}).get("passenger_capacity", self.instance.get("bus", {}).get("passenger_capacity", 80))), "onboard_parcel_ids": [], "active": True, "completed": False, "accumulated_delay_min": 0.0, "accumulated_operating_delay_min": 0.0}
        self.bus_states = {tid: self._bus_for_trip(tid) for tid in self.trip_ids}
        for t in self.instance["network"]["scheduled_bus_trips"]:
            tid = int(t["trip_id"])
            bus = self._bus_for_trip(tid)
            bus["trip_id"] = tid
            bus["onboard_parcel_ids"] = [int(d["customer_id"]) for d in self.assignment.get("decisions", []) if int(d["trip_id"]) == tid]
            bus["next_arrival_time_min"] = float(t["departure_min"])
            self.calendar.add_bus_arrival(time=float(t["departure_min"]), trip_id=tid, stop_index=0, station_id=self.station_by_stop.get(self.stop_ids[0], -1), integrated=self.stop_ids[0] in self.station_by_stop, passengers_required=False, parcel_required=False)

        self.station_states = {}
        for s in self.instance["stations"]["stations"]:
            sid = int(s["station_id"])
            n_ch = int(s["chargers"])
            station_load_profile = self.scenario.get("power", {}).get("station_base_load_kw", {}).get(str(sid), self.scenario.get("power", {}).get("station_base_load_kw", {}).get(sid, []))
            self.station_states[sid] = {"station_id": sid, "charger_release_times_min": [0.0] * n_ch, "bus_charging_intervals": [], "locker_parcels": [], "locker_parcel_ids": [], "locker_inventory_kg": 0.0, "locker_capacity_kg": float(s["locker_capacity_kg"]), "idle_drone_ids": [f"s{sid}_d{j}" for j in range(int(s["drones"]))], "active_drone_missions": [], "pending_parcel_releases": [], "full_batteries": int(s["initial_fully_charged_batteries"]), "depleted_batteries": int(s["initial_depleted_batteries"]), "batteries_charging": [], "charging_batteries": [], "current_bus_charging_load": 0.0, "current_drone_battery_charging_load": 0.0, "power_capacity_kw": float(s["station_power_capacity_kw"]), "charging_slots": n_ch, "G_max": int(self.instance.get("battery", {}).get("max_simultaneous_charging", 6)), "P_capacity": float(s["station_power_capacity_kw"]), "P_bat": float(self.instance.get("battery", {}).get("charge_power_kw", 2.0)), "P_chg": float(self.instance["charging"]["pantograph_power_kw"]), "drones": [{"drone_id": f"s{sid}_d{j}", "status": "idle"} for j in range(int(s["drones"]))], "battery_charge_duration_min": float(self.instance.get("battery", {}).get("charge_duration_min", 45.0)), "battery_capacity_kwh": float(self.instance.get("battery", {}).get("capacity_kwh", 0.0)), "dispatch_interval_min": float(self.config.get("drone", {}).get("dispatch_interval_min", 5.0)), "max_round_trip_duration_min": float(self.config.get("drone", {}).get("max_round_trip_duration_min", 1e9)), "base_load_profile_kw": station_load_profile, "base_load_fallback_kw": 50.0}

        self.parcel_states = {}
        customers = {int(c["customer_id"]): c for c in self.instance.get("customers", [])}
        for cid, t_id in self.assignment_index.get("by_customer", {}).items():
            c = customers.get(int(cid), {})
            assigned_station = int(self.assignment_index["station_by_customer"][cid])
            option = next((o for o in c.get("feasible_stations", []) if int(o["station_id"]) == assigned_station), None)
            if option is None:
                raise ValueError(f"Missing feasible station option for customer {cid} at station {assigned_station}")
            rt_min = float(option["mission_duration_min"])
            if "outbound_with_service_time_min" in option:
                out_min = float(option["outbound_with_service_time_min"])
            else:
                dist_km = float(option.get("distance_km", 0.0))
                speed_kmh = float(self.instance["drone"]["speed_kmh"])
                svc_min = float(self.instance["drone"]["customer_service_time_min"])
                out_min = (dist_km / speed_kmh) * 60.0 + svc_min
            self.parcel_states[cid] = {
                "parcel_id": int(cid), "id": int(cid), "customer_id": int(cid),
                "weight_kg": float(c.get("parcel_weight_kg", 0.0)),
                "delivery_deadline_min": float(c.get("delivery_deadline_min", self.horizon)),
                "deadline_min": float(c.get("delivery_deadline_min", self.horizon)),
                "assigned_trip_id": int(t_id), "assigned_station_id": assigned_station,
                "status": "onboard", "release_time_min": None, "pickup_time_min": None,
                "delivery_completion_time_min": None, "drone_return_time_min": None,
                "locker_holding_time_min": None, "T_out_min": out_min, "T_rt_min": rt_min,
                "T_out": out_min, "T_rt": rt_min,
                "drone_cost": float(self.instance["parcel"]["cost"]["drone_per_km"]) * float(option["distance_km"]),
            }

        self.delivered_parcels = set()
        self.state = {"time": 0.0, "horizon": self.horizon, "trip_location": 0, "battery": cap, "battery_max": cap, "onboard_passengers": 0, "onboard_parcels": 0, "queue": 0, "locker": 0, "idle_drones": 0, "full_batteries": 0, "station_power": 0.0, "power_margin": 0.0, "available_chargers": 0, "parcel_urgency": 0.0, "calendar_len": len(self.calendar), "charge_power_kw": float(self.instance["charging"]["pantograph_power_kw"]), "charge_efficiency": float(self.instance["charging"].get("charger_efficiency", 0.95)), "travel_energy_kwh_per_km": float(self.instance["bus"]["energy_kwh_per_km"]), "travel_distance_km": float(self.instance["network"].get("interstop_distance_km", 1.0)), "trip_id": 0, "bus_operation_horizon": self.bus_operation_horizon, "delivery_evaluation_horizon": self.delivery_evaluation_horizon}
        self.obs_schema = {"trip_ids": self.trip_ids, "station_ids": self.station_ids, "stop_ids": self.stop_ids, "battery_capacity_kwh": cap, "passenger_capacity": float(self.config.get("bus", {}).get("passenger_capacity", 80)), "freight_capacity_kg": float(self.config.get("bus", {}).get("freight_capacity_kg", 20.0)), "locker_capacity_kg": max(float(s["locker_capacity_kg"]) for s in self.instance["stations"]["stations"]), "drones_per_station": max(int(s["drones"]) for s in self.instance["stations"]["stations"]), "chargers_per_station": max(int(s["chargers"]) for s in self.instance["stations"]["stations"]), "station_power_capacity_kw": max(float(s["station_power_capacity_kw"]) for s in self.instance["stations"]["stations"]), "battery_inv_norm": float(max(1, self.instance.get("battery", {}).get("initial_fully_charged_per_station", 6))), "urgency_count_norm": 20.0, "queue_norm": 100.0, "horizon": self.horizon}
        self.observation_feature_names = build_feature_names(self.obs_schema)


    def _parcel_delivered_by_time(self, parcel: dict, t_eval: float) -> bool:
        completion = parcel.get("delivery_completion_time_min")
        return completion is not None and float(completion) <= float(t_eval) + 1e-9

    def _terminal_undelivered_parcels(self, t_eval: float) -> list[dict]:
        return [p for p in self.parcel_states.values() if not self._parcel_delivered_by_time(p, t_eval)]

    def _available_chargers(self, station_state, now):
        return sum(1 for t in station_state["charger_release_times_min"] if t <= now)
    def _active_bus_chargers(self, station_state, now):
        intervals = station_state.get("bus_charging_intervals", [])
        return sum(1 for it in intervals if float(it.get("start_time_min", 0.0)) <= now < float(it.get("end_time_min", 0.0)))
    def _active_bus_charging_load_kw(self, station_state, now):
        intervals = station_state.get("bus_charging_intervals", [])
        return sum(float(it.get("charging_power_kw", 0.0)) for it in intervals if float(it.get("start_time_min", 0.0)) <= now < float(it.get("end_time_min", 0.0)))
    def _base_load_kw(self, station_state, now):
        """Piecewise-constant base load sampled on a 1-minute grid (left-closed, right-open)."""
        prof = station_state.get("base_load_profile_kw", [])
        if prof:
            i = max(0, min(int(float(now)), len(prof) - 1))
            return float(prof[i])
        return float(station_state.get("base_load_fallback_kw", 50.0))

    def _next_base_load_change_time(self, station_state, after_t: float) -> float | None:
        prof = station_state.get("base_load_profile_kw", [])
        if not prof:
            return None
        next_idx = int(float(after_t)) + 1
        if 0 <= next_idx < len(prof):
            return float(next_idx)
        return None

    def _run_station_interval(self, start_t: float, end_t: float):
        if end_t <= start_t:
            return
        slice_min = float(self.config.get("env", {}).get("station_update_slice_min", 1.0))
        if slice_min <= 0:
            slice_min = 1.0
        t = float(start_t)
        def _next_low_level_event_time(after_t: float) -> float | None:
            nxt = None
            for st in self.station_states.values():
                for rel in st.get("pending_parcel_releases", []):
                    rt = float(rel.get("release_time_min", 0.0))
                    if rt > after_t + 1e-9 and (nxt is None or rt < nxt):
                        nxt = rt
                for interval in st.get("bus_charging_intervals", []):
                    et = float(interval.get("end_time_min", 0.0))
                    if et > after_t + 1e-9 and (nxt is None or et < nxt):
                        nxt = et
                bl_next = self._next_base_load_change_time(st, after_t)
                if bl_next is not None and bl_next > after_t + 1e-9 and (nxt is None or bl_next < nxt):
                    nxt = bl_next
                for m in st.get("active_drone_missions", []):
                    rt = float(m.get("eta_return_min", m.get("drone_return_time", 0.0)))
                    if rt > after_t + 1e-9 and (nxt is None or rt < nxt):
                        nxt = rt
                for job in st.get("charging_batteries", []):
                    ct = float(job.get("completion_time_min", 0.0))
                    if ct > after_t + 1e-9 and (nxt is None or ct < nxt):
                        nxt = ct
            for p in self.parcel_states.values():
                ct = p.get("delivery_completion_time_min", p.get("delivery_completion_time"))
                if p.get("status") == "assigned_to_drone" and ct is not None:
                    ctf = float(ct)
                    if ctf > after_t + 1e-9 and (nxt is None or ctf < nxt):
                        nxt = ctf
            return nxt
        while t < end_t - 1e-9:
            t_next = min(end_t, t + slice_min)
            next_event = _next_low_level_event_time(t)
            if next_event is not None:
                t_next = min(t_next, next_event)
            for st in self.station_states.values():
                st["last_power_update_time_min"] = min(float(st.get("last_power_update_time_min", t)), float(t))
                p_e = self._active_bus_charging_load_kw(st, t)
                p_l = self._base_load_kw(st, t)
                st["current_bus_charging_load"] = p_e
                active_bus_chargers = self._active_bus_chargers(st, t)
                dt = max(0.0, t_next - t)
                st["charger_occupied_time_min"] = float(st.get("charger_occupied_time_min", 0.0)) + active_bus_chargers * dt
                st["charger_observation_time_min"] = float(st.get("charger_observation_time_min", 0.0)) + max(1, len(st.get("charger_release_times_min", []))) * dt
                denom = max(1e-9, float(st["charger_observation_time_min"]))
                st["charger_utilization"] = float(st["charger_occupied_time_min"]) / denom
                released = []
                remaining = []
                for rel in st.get("pending_parcel_releases", []):
                    if float(rel.get("release_time_min", 0.0)) <= t_next + 1e-9:
                        released.extend(unload_parcels_to_locker(
                            self._bus_for_trip(int(rel["trip_id"])),
                            st,
                            self.parcel_states,
                            list(rel.get("parcel_ids", [])),
                            float(rel.get("release_time_min", t_next)),
                        ))
                    else:
                        remaining.append(rel)
                st["pending_parcel_releases"] = remaining
                operate_station_step(
                    st,
                    t_next,
                    parcel_states=self.parcel_states,
                    delivered_parcels=self.delivered_parcels,
                    p_e=p_e,
                    p_l=p_l,
                    new_parcels=bool(released),
                    dispatch_interval=float(st.get("dispatch_interval_min", 5.0)),
                    max_round_trip_duration=float(st.get("max_round_trip_duration_min", 1e9)),
                )
            t = t_next

    def _advance_until_decision(self):
        self.current_decision_event = None
        self.current_event = None
        while len(self.calendar) > 0:
            next_event = self.calendar.peek_next()
            if next_event is not None and float(next_event.time) > self.bus_operation_horizon + 1e-9:
                prev_t = float(self.state.get("time", 0.0))
                cutoff_t = float(self.delivery_evaluation_horizon)
                if cutoff_t > prev_t:
                    self._run_station_interval(prev_t, cutoff_t)
                self.state["time"] = cutoff_t
                self.current_event = None
                return
            e = self.calendar.pop_next()
            prev_t = float(self.state.get("time", 0.0))
            self._run_station_interval(prev_t, float(e.time))
            self.state["time"] = float(e.time)
            if e.kind == "charger_release":
                continue
            if e.kind == "dispatch_tick":
                st = self.station_states[e.station_id]
                p_e = self._active_bus_chargers(st, self.state["time"]) * self.state["charge_power_kw"]
                p_l = self._base_load_kw(st, self.state["time"])
                st["current_bus_charging_load"] = p_e
                operate_station_step(st, self.state["time"], parcel_states=self.parcel_states, delivered_parcels=self.delivered_parcels, p_e=p_e, p_l=p_l, new_parcels=False, dispatch_interval=float(st.get("dispatch_interval_min", 5.0)), max_round_trip_duration=float(st.get("max_round_trip_duration_min", 1e9)))
                continue
            if e.kind == "bus_arrival":
                if self._process_bus_arrival(e):
                    self.current_decision_event = e
                    self.current_event = e
                    return

    def _process_bus_arrival(self, e):
        bus = self._bus_for_trip(int(e.trip_id))
        idx = int(e.stop_index)
        bus["trip_id"] = int(e.trip_id)
        if idx == 0:
            bus["completed"] = False
            bus["active"] = True
            bus["onboard_parcel_ids"] = [int(d["customer_id"]) for d in self.assignment.get("decisions", []) if int(d["trip_id"]) == int(e.trip_id)]
        stop_id = self.stop_ids[idx]
        bus["current_stop_index"] = idx
        bus["next_stop_id"] = self.stop_ids[idx + 1] if idx + 1 < len(self.stop_ids) else -1

        station_id = self.station_by_stop.get(stop_id, -1)
        integrated = station_id != -1
        pax_required = False
        parcel_ids = get_unloading_parcels(bus["trip_id"], station_id, self.assignment_index, self.parcel_states) if (integrated and int(bus["trip_id"]) in self.freight_carrying_trip_ids) else []
        parcel_required = len(parcel_ids) > 0
        rate = self._arrival_rate(stop_id, float(e.time))
        al_p = self._alighting_prob(stop_id, float(e.time))
        elapsed = max(0.0, float(e.time) - float(self.stop_last_update.get(stop_id, 0.0)))
        arrivals = sample_poisson_arrivals(rate, elapsed, self.rng)
        queue_at_arrival = self.stop_queues[stop_id] + arrivals
        onboard_at_arrival = int(bus["onboard_passengers"])
        service_preview = sample_initial_passenger_event(queue=queue_at_arrival, onboard=onboard_at_arrival, capacity=bus["passenger_capacity"], alighting_probability=al_p, rho_al_min_per_pax=1.5/60.0, rho_bo_min_per_pax=3.0/60.0, rng=self.rng)
        pax_required = bool(service_preview["chi"])
        requires_stop = pax_required or parcel_required

        if requires_stop and integrated:
            e.station_id, e.integrated, e.passengers_required, e.parcel_required, e.requires_stop, e.is_decision = station_id, True, pax_required, parcel_required, True, True
            e.passenger_service_preview = service_preview
            e.arrival_queue_before_preview = int(queue_at_arrival)
            e.arrival_onboard_before_preview = int(onboard_at_arrival)
            e.unloading_volume_kg = compute_unloading_volume(parcel_ids, self.parcel_states)
            return True

        if requires_stop:
            service = finalize_passenger_service(initial_event=service_preview, capacity=bus["passenger_capacity"], rate_per_min=rate, rho_bo_min_per_pax=3.0/60.0, parcel_unloading_time_min=0.0, charging_duration_min=0.0, rng=self.rng)
            bus["onboard_passengers"] = service["onboard_final"]
            self.stop_queues[stop_id] = service["queue_final"]
            self.stop_last_update[stop_id] = float(e.time) + float(service["realized_dwell_min"])
            dwell = service["realized_dwell_min"]
        else:
            self.stop_queues[stop_id] = int(queue_at_arrival)
            self.stop_last_update[stop_id] = float(e.time)
            dwell = 0.0
        dep = e.time + dwell
        if idx + 1 < len(self.stop_ids):
            next_t = dep + float(self.travel_times[idx + 1] - self.travel_times[idx])
            self.calendar.add_bus_arrival(time=next_t, trip_id=bus["trip_id"], stop_index=idx + 1, station_id=self.station_by_stop.get(self.stop_ids[idx+1], -1), integrated=self.stop_ids[idx+1] in self.station_by_stop, passengers_required=False, parcel_required=False)
            bus["next_arrival_time_min"] = next_t
            bus["battery_kwh"] = apply_travel_consumption(bus["battery_kwh"], self.state["travel_distance_km"], self.state["travel_energy_kwh_per_km"])
        else:
            self._apply_non_service_return(bus)
            bus["active"] = False
        return False


    def _lookup_stop_time_value(self, per_stop_time: dict, stop_id: int, time_min: float, default: float = 0.0) -> float:
        series = per_stop_time.get(str(stop_id), per_stop_time.get(stop_id, None)) if isinstance(per_stop_time, dict) else None
        if not isinstance(series, list) or not series:
            return float(default)
        idx = max(0, min(int(time_min), len(series)-1))
        return float(series[idx])

    def _arrival_rate(self, stop_id: int, time_min: float) -> float:
        pax = self.scenario.get("passenger", {})
        prof = pax.get("arrival_rate_profile_per_stop_per_min", {})
        if prof:
            return max(0.0, self._lookup_stop_time_value(prof, stop_id, time_min, 0.0))
        return max(0.0, float(pax.get("arrival_rate_per_stop_per_min", {}).get(str(stop_id), pax.get("arrival_rate_per_stop_per_min", {}).get(stop_id, 0.0))))

    def _alighting_prob(self, stop_id: int, time_min: float) -> float:
        pax = self.scenario.get("passenger", {})
        prof = pax.get("alighting_profile_per_stop", {})
        if prof:
            return min(1.0, max(0.0, self._lookup_stop_time_value(prof, stop_id, time_min, 0.0)))
        return min(1.0, max(0.0, float(pax.get("alighting_probability", 0.0))))

    def get_action_mask(self) -> np.ndarray:
        if self.current_decision_event is None:
            z = np.zeros(len(self.action_set), dtype=np.int32)
            if len(z) > 0:
                z[0] = 1
            return z
        bus = self._bus_for_trip(int(self.current_decision_event.trip_id))
        st = self.station_states[int(self.current_decision_event.station_id)]
        return feasible_action_mask(self._available_chargers(st, self.state["time"]), bus["battery_kwh"], bus["battery_capacity_kwh"], self.state["charge_power_kw"], self.state["charge_efficiency"], action_set=self.action_set, max_single_stop_seconds=float(self.config.get("charging", {}).get("max_single_stop_seconds", max(self.action_set))))
    def get_feasible_actions(self): return feasible_actions(self.get_action_mask())
    def repair_action(self, action_index): return repair_action(action_index, self.get_action_mask())


    def _reward_alphas(self) -> dict:
        return self.config.get("reward", {"alpha_1": 1.0, "alpha_2": 1.0, "alpha_3": 0.2, "alpha_4": 1.0, "alpha_5": 5.0, "alpha_6": 1.0})


    def _snapshot_cumulative_metrics(self) -> dict:
        now = float(self.state.get("time", 0.0))
        delivered = [p for p in self.parcel_states.values() if self._parcel_delivered_by_time(p, now)]
        parcel_lateness = sum(max(0.0, float(p["delivery_completion_time_min"]) - float(p.get("delivery_deadline_min", p.get("deadline_min", p["delivery_completion_time_min"])))) for p in delivered)
        late_delivery_count = sum(1 for p in delivered if float(p["delivery_completion_time_min"]) > float(p.get("delivery_deadline_min", p.get("deadline_min", p["delivery_completion_time_min"]))))
        energy_bus = sum(float(st.get("bus_charging_energy_kwh", 0.0)) for st in self.station_states.values())
        energy_drone = sum(float(st.get("drone_charging_energy_kwh", 0.0)) for st in self.station_states.values())
        power_overload = sum(float(st.get("power_overload_amount_kw_min", 0.0)) for st in self.station_states.values())
        max_power_overload = max((float(st.get("maximum_power_overload_kw", 0.0)) for st in self.station_states.values()), default=0.0)
        locker_overflow = sum(float(st.get("locker_overflow_amount_kg_min", 0.0)) for st in self.station_states.values())
        power_overload_duration = sum(float(st.get("power_overload_duration_min", 0.0)) for st in self.station_states.values())
        locker_overflow_duration = sum(float(st.get("locker_overflow_duration_min", 0.0)) for st in self.station_states.values())
        locker_overflow_amount = sum(float(st.get("locker_overflow_amount_kg", 0.0)) for st in self.station_states.values())
        return {
            "passenger_delay": sum(float(b.get("accumulated_delay_min", 0.0)) for b in self.bus_states.values()),
            "bus_operating_delay": sum(float(b.get("accumulated_operating_delay_min", 0.0)) for b in self.bus_states.values()),
            "parcel_lateness": float(parcel_lateness),
            "late_delivery_count": float(late_delivery_count),
            "delivered_count": float(len(delivered)),
            "energy_consumption": float(energy_bus + energy_drone),
            "power_overload": float(power_overload),
            "maximum_station_power_overload": float(max_power_overload),
            "battery_violation": sum(max(0.0, float(b.get("safety_battery_kwh", 0.0)) - float(b.get("battery_kwh", 0.0))) for b in self.bus_states.values()),
            "locker_overflow": float(locker_overflow),
            "bus_charging_energy_kwh": float(energy_bus),
            "drone_charging_energy_kwh": float(energy_drone),
            "power_overload_duration": float(power_overload_duration),
            "locker_overflow_duration": float(locker_overflow_duration),
            "locker_overflow_amount": float(locker_overflow_amount),
        }

    def _build_transition_reward(self, before: dict, after: dict, terminal_penalty: float) -> tuple[float, dict]:
        required = ["passenger_delay", "parcel_lateness", "late_delivery_count", "delivered_count", "energy_consumption", "power_overload", "battery_violation", "locker_overflow"]
        missing = [k for k in required if k not in before or k not in after]
        if missing:
            raise ValueError(f"Missing cumulative metrics for reward delta: {missing}")
        passenger_delta = max(0.0, float(after["passenger_delay"] - before["passenger_delay"]))
        parcel_delta = max(0.0, float(after["parcel_lateness"] - before["parcel_lateness"]))
        late_delivery_delta = max(0.0, float(after["late_delivery_count"] - before["late_delivery_count"]))
        delivered_count_delta = max(0.0, float(after["delivered_count"] - before["delivered_count"]))
        energy_delta = max(0.0, float(after["energy_consumption"] - before["energy_consumption"]))
        power_delta = max(0.0, float(after["power_overload"] - before["power_overload"]))
        battery_delta = max(0.0, float(after["battery_violation"] - before["battery_violation"]))
        locker_delta = max(0.0, float(after["locker_overflow"] - before["locker_overflow"]))
        eta_e = float(self.config.get("reward", {}).get("eta_E", 1.0))
        eta_p = float(self.config.get("reward", {}).get("eta_P", 1.0))
        components = {
            "passenger_delay": passenger_delta,
            "parcel_lateness": parcel_delta + float(terminal_penalty),
            "energy_cost": energy_delta * eta_e,
            "power_overload": power_delta * eta_p,
            "battery_safety": battery_delta,
            "locker_overflow": locker_delta,
            "terminal_penalty": float(terminal_penalty),
            "bus_charging_energy_kwh": max(0.0, float(after["bus_charging_energy_kwh"] - before["bus_charging_energy_kwh"])),
            "drone_charging_energy_kwh": max(0.0, float(after["drone_charging_energy_kwh"] - before["drone_charging_energy_kwh"])),
            "total_energy_kwh": energy_delta,
            "power_overload_duration": max(0.0, float(after["power_overload_duration"] - before["power_overload_duration"])),
            "locker_overflow_duration": max(0.0, float(after["locker_overflow_duration"] - before["locker_overflow_duration"])),
            "locker_overflow_amount": max(0.0, float(after["locker_overflow_amount"] - before["locker_overflow_amount"])),
            "number_late_deliveries": late_delivery_delta,
            "late_delivery_count_delta": late_delivery_delta,
            "delivered_count_delta": delivered_count_delta,
        }
        return compute_reward(components, self._reward_alphas())

    def step(self, action_index):
        e = self.current_decision_event
        if e is None:
            self.current_event = None
            term, reason = check_termination(self.state, False)
            return self._build_obs_for_current_event(), 0.0, term, False, {"termination_reason": reason}
        bus = self._bus_for_trip(int(e.trip_id))
        st = self.station_states[int(e.station_id)]
        mask = self.get_action_mask()
        if action_index < 0 or action_index >= len(mask):
            raise IndexError(f"Action index {action_index} out of bounds for action_dim={len(mask)}")
        invalid_action = bool(mask[action_index] == 0)
        strict_invalid = bool(self.config.get("env", {}).get("strict_action_validation", False))
        if invalid_action and strict_invalid:
            raise ValueError(
                f"Infeasible action index {action_index} at time={self.state['time']:.3f}. "
                f"Feasible actions={self.get_feasible_actions()}"
            )
        # Deterministic infeasible-action repair rule: when the requested action is infeasible,
        # execute the largest feasible action index that does not exceed the requested index.
        # If no positive duration is feasible (e.g., no charger available), this resolves to u=0.
        ex_idx = action_index if not invalid_action else self.repair_action(action_index)
        feasible_count = int(np.sum(mask))
        repair_reason = "none"
        if invalid_action:
            self.episode_metrics["invalid_action_count"] = int(self.episode_metrics.get("invalid_action_count", 0)) + 1
            self.episode_metrics["action_repair_count"] = int(self.episode_metrics.get("action_repair_count", 0)) + 1
            repair_reason = "infeasible_action_repaired_to_nearest_feasible_not_exceeding_request"
        self.episode_metrics["requested_action_sum"] = float(self.episode_metrics.get("requested_action_sum", 0.0)) + float(action_index)
        self.episode_metrics["executed_action_sum"] = float(self.episode_metrics.get("executed_action_sum", 0.0)) + float(ex_idx)
        self.episode_metrics["action_gap_sum"] = float(self.episode_metrics.get("action_gap_sum", 0.0)) + abs(float(action_index) - float(ex_idx))
        requested_dur = action_index_to_duration(action_index, self.action_set)
        dur = action_index_to_duration(ex_idx, self.action_set)
        requested_duration_min = float(requested_dur) / 60.0
        executed_duration_min = float(dur) / 60.0
        self.episode_metrics["requested_duration_min_sum"] = float(self.episode_metrics.get("requested_duration_min_sum", 0.0)) + requested_duration_min
        self.episode_metrics["executed_duration_min_sum"] = float(self.episode_metrics.get("executed_duration_min_sum", 0.0)) + executed_duration_min
        if executed_duration_min > 0.0:
            self.episode_metrics["charging_event_count"] = float(self.episode_metrics.get("charging_event_count", 0.0)) + 1.0
            self.episode_metrics["positive_executed_duration_min_sum"] = float(self.episode_metrics.get("positive_executed_duration_min_sum", 0.0)) + executed_duration_min
        else:
            self.episode_metrics["zero_charging_event_count"] = float(self.episode_metrics.get("zero_charging_event_count", 0.0)) + 1.0
        before = bus["battery_kwh"]
        av_before = self._available_chargers(st, self.state["time"])
        if dur > 0 and av_before > 0:
            i = next(i for i, t in enumerate(st["charger_release_times_min"]) if t <= self.state["time"])
            st["charger_release_times_min"][i] = self.state["time"] + dur / 60.0
            st.setdefault("bus_charging_intervals", []).append({
                "station_id": int(st["station_id"]),
                "bus_id": bus["vehicle_id"],
                "charger_index": int(i),
                "start_time_min": float(self.state["time"]),
                "end_time_min": float(st["charger_release_times_min"][i]),
                "charging_power_kw": float(self.state["charge_power_kw"]),
            })
            self.calendar.add_charger_release(time=st["charger_release_times_min"][i], station_id=st["station_id"])
            st["current_bus_charging_load"] = self._active_bus_charging_load_kw(st, self.state["time"])
        bus["battery_kwh"] = apply_charge(bus["battery_kwh"], dur, self.state["charge_power_kw"], self.state["charge_efficiency"], bus["battery_capacity_kwh"])

        unload_ids = get_unloading_parcels(bus["trip_id"], st["station_id"], self.assignment_index, self.parcel_states) if int(bus["trip_id"]) in self.freight_carrying_trip_ids else []
        qf = compute_unloading_volume(unload_ids, self.parcel_states)
        unloading_time_per_kg_min = float(self.instance.get("parcel", {}).get("unloading_time_sec_per_kg", 30.0)) / 60.0
        stop_id = self.stop_ids[int(e.stop_index)]
        rate = self._arrival_rate(stop_id, float(e.time))
        al_p = self._alighting_prob(stop_id, float(e.time))
        bus["onboard_before_service"] = bus["onboard_passengers"]
        bus["battery_before_step"] = bus["battery_kwh"]
        initial_event = getattr(e, "passenger_service_preview", None) or sample_initial_passenger_event(queue=int(getattr(e, "arrival_queue_before_preview", self.stop_queues[stop_id])), onboard=int(getattr(e, "arrival_onboard_before_preview", bus["onboard_passengers"])), capacity=bus["passenger_capacity"], alighting_probability=al_p, rho_al_min_per_pax=1.5/60.0, rho_bo_min_per_pax=3.0/60.0, rng=self.rng)
        service = finalize_passenger_service(initial_event=initial_event, capacity=bus["passenger_capacity"], rate_per_min=rate, rho_bo_min_per_pax=3.0/60.0, parcel_unloading_time_min=qf * unloading_time_per_kg_min, charging_duration_min=dur/60.0, rng=self.rng)
        bus["onboard_passengers"] = service["onboard_final"]
        self.stop_queues[stop_id] = service["queue_final"]
        self.stop_last_update[stop_id] = float(self.state["time"]) + float(service["realized_dwell_min"])
        release_time = float(self.state["time"]) + float(qf * unloading_time_per_kg_min)
        if unload_ids:
            st.setdefault("pending_parcel_releases", []).append({
                "trip_id": int(bus["trip_id"]),
                "parcel_ids": list(unload_ids),
                "release_time_min": release_time,
            })
        unloaded = []

        raw_dep = self.state["time"] + service["realized_dwell_min"]
        dep = min(float(raw_dep), float(self.bus_operation_horizon))
        idx = int(e.stop_index)
        if idx + 1 < len(self.stop_ids) and dep < self.bus_operation_horizon - 1e-9:
            next_t = dep + float(self.travel_times[idx + 1] - self.travel_times[idx])
            self.calendar.add_bus_arrival(time=next_t, trip_id=bus["trip_id"], stop_index=idx + 1, station_id=self.station_by_stop.get(self.stop_ids[idx+1], -1), integrated=self.stop_ids[idx+1] in self.station_by_stop, passengers_required=False, parcel_required=False)
            bus["next_arrival_time_min"] = next_t
            bus["battery_kwh"] = apply_travel_consumption(bus["battery_kwh"], self.state["travel_distance_km"], self.state["travel_energy_kwh_per_km"])
        else:
            self._apply_non_service_return(bus)
            bus["active"] = False

        self._run_station_interval(float(self.state["time"]), dep)
        p_e = self._active_bus_charging_load_kw(st, dep)
        p_l = self._base_load_kw(st, dep)
        st["current_bus_charging_load"] = p_e
        op = operate_station_step(st, dep, parcel_states=self.parcel_states, delivered_parcels=self.delivered_parcels, p_e=p_e, p_l=p_l, new_parcels=bool(unloaded), dispatch_interval=float(st.get("dispatch_interval_min", 5.0)), max_round_trip_duration=float(st.get("max_round_trip_duration_min", 1e9)))
        passenger_dwell_min = float(service.get("passenger_dwell_min", 0.0))
        freight_dwell_min = float(qf * unloading_time_per_kg_min)
        charging_duration_min = float(dur) / 60.0
        realized_dwell_min = float(service.get("realized_dwell_min", 0.0))
        has_passenger_service = bool(service.get("chi", False))
        additional_dwell_min = max(0.0, realized_dwell_min - passenger_dwell_min) if has_passenger_service else max(0.0, realized_dwell_min)
        affected_passengers = int(service.get("affected_passengers", service.get("onboard_final", 0)))
        event_passenger_delay = float(max(0, affected_passengers) * additional_dwell_min)
        bus["accumulated_delay_min"] = float(bus.get("accumulated_delay_min", 0.0)) + event_passenger_delay
        bus["accumulated_operating_delay_min"] = float(bus.get("accumulated_operating_delay_min", 0.0)) + additional_dwell_min
        self.event_log.append({"time_min": self.state["time"], "bus_id": bus["vehicle_id"], "trip_id": bus["trip_id"], "station_id": st["station_id"], "action_sec": dur, "dwell_time_min": realized_dwell_min, "passenger_delay": event_passenger_delay, "parcel_unloading_kg": qf, "battery_before": before, "battery_after_charge": apply_charge(before, dur, self.state["charge_power_kw"], self.state["charge_efficiency"], bus["battery_capacity_kwh"]), "available_chargers_before": av_before, "station_power_overload": float(op["power"].get("overload", 0.0))})

        transition_start_time = float(self.state.get("time", 0.0))
        self.state.update({"time": dep, "trip_id": bus["trip_id"], "trip_location": idx, "battery": bus["battery_kwh"], "battery_max": bus["battery_capacity_kwh"], "onboard_passengers": bus["onboard_passengers"], "onboard_parcels": len(bus["onboard_parcel_ids"]), "locker": int(st["locker_inventory_kg"]), "idle_drones": sum(1 for d in st.get("drones", []) if d.get("status") == "idle"), "full_batteries": st.get("full_batteries", 0), "station_power": op["P_tot"], "power_margin": st["power_capacity_kw"] - op["P_tot"], "available_chargers": self._available_chargers(st, dep), "calendar_len": len(self.calendar)})
        before_metrics = self._snapshot_cumulative_metrics()
        before_metrics["passenger_delay"] -= event_passenger_delay
        before_metrics["bus_operating_delay"] = float(before_metrics.get("bus_operating_delay", 0.0)) - additional_dwell_min
        self.episode_metrics["number_decision_events"] += 1
        self._advance_until_decision()
        self.state["time"] = min(float(self.state.get("time", 0.0)), float(self.delivery_evaluation_horizon))
        if self.state["time"] > self.delivery_evaluation_horizon + 1e-6:
            raise RuntimeError(f"Environment time exceeded evaluation horizon: {self.state['time']} > {self.delivery_evaluation_horizon}")
        terminated, reason = check_termination(self.state, self.current_decision_event is not None)
        terminal_penalty = 0.0
        terminal_eval_time = float(self.delivery_evaluation_horizon if reason == "horizon_reached" else self.state["time"])
        if terminated:
            undelivered_terminal = self._terminal_undelivered_parcels(terminal_eval_time)
            terminal_penalty = apply_terminal_penalty_once(self.__dict__, undelivered_terminal, terminal_eval_time, float(self.config.get("reward", {}).get("eta_l_term", 1.0)), float(self.config.get("reward", {}).get("eta_u_term", 1.0)))
            self.episode_metrics["terminal_penalty"] += terminal_penalty
        after_metrics = self._snapshot_cumulative_metrics()
        reward, rc = self._build_transition_reward(before_metrics, after_metrics, terminal_penalty)
        sums = self.episode_metrics.setdefault("reward_component_sums", {})
        for k, v in rc.items():
            if isinstance(v, (int, float)):
                sums[k] = float(sums.get(k, 0.0)) + float(v)
        undelivered_terminal_count = float(len(self._terminal_undelivered_parcels(terminal_eval_time))) if terminated else 0.0
        if terminated and reason == "horizon_reached" and abs(float(self.state.get("time", 0.0)) - float(self.delivery_evaluation_horizon)) > 1e-6:
            raise RuntimeError(f"Horizon termination must end exactly at horizon: t={self.state.get('time')} horizon={self.delivery_evaluation_horizon}")
        reported_unload_ids = unload_ids if int(bus["trip_id"]) in self.freight_carrying_trip_ids else []
        reported_qf = qf if reported_unload_ids else 0.0
        return self._build_obs_for_current_event(), float(reward), terminated, False, {"executed_action_index": ex_idx, "executed_duration": dur, "requested_duration_min": requested_duration_min, "executed_duration_min": executed_duration_min, "selected_action": int(action_index), "requested_action": int(action_index), "executed_action": int(ex_idx), "action_repaired": ex_idx != action_index, "was_action_repaired": ex_idx != action_index, "invalid_action": invalid_action, "invalid_action_count": int(self.episode_metrics.get("invalid_action_count", 0)), "action_repair_count": int(self.episode_metrics.get("action_repair_count", 0)), "repair_reason": repair_reason, "feasible_action_count": feasible_count, "termination_reason": reason, "reward_components": rc, "event": e, "unloaded_parcels": reported_unload_ids, "unloading_volume_kg": reported_qf, "unloading_duration_min": reported_qf * unloading_time_per_kg_min, "parcel_release_time_min": release_time, "current_trip_id": bus["trip_id"], "current_bus_id": bus["vehicle_id"], "current_station_id": st["station_id"], "transition_start_time": transition_start_time, "transition_end_time": float(self.state.get("time", dep)), "passenger_delay_delta": rc.get("passenger_delay", 0.0), "parcel_lateness_delta": rc.get("parcel_lateness", 0.0) - rc.get("terminal_penalty", 0.0), "late_delivery_count_delta": rc.get("late_delivery_count_delta", 0.0), "delivered_count_delta": rc.get("delivered_count_delta", 0.0), "undelivered_terminal_count": undelivered_terminal_count, "energy_consumption_delta": rc.get("total_energy_kwh", 0.0), "power_overload_delta": rc.get("power_overload", 0.0), "battery_violation_delta": rc.get("battery_safety", 0.0), "locker_overflow_delta": rc.get("locker_overflow", 0.0), "terminal_undelivered_penalty": rc.get("terminal_penalty", 0.0), "feasible_action_mask": mask.tolist(), "passenger_service": service, "dwell_components": {"passenger_dwell_min": passenger_dwell_min, "freight_dwell_min": freight_dwell_min, "charging_duration_min": charging_duration_min, "realized_dwell_min": realized_dwell_min, "additional_dwell_min": additional_dwell_min, "affected_passengers": affected_passengers, "event_passenger_delay": event_passenger_delay}, "departure_time_min": dep, "bus_operating_delay_delta": additional_dwell_min}


    def get_episode_metrics(self) -> dict:
        cumulative = self._snapshot_cumulative_metrics()
        reward_sums = dict(self.episode_metrics.get("reward_component_sums", {}))
        decision_events = int(self.episode_metrics.get("number_decision_events", 0))
        avg_excess = (cumulative["bus_operating_delay"] / decision_events) if decision_events > 0 else 0.0
        metrics = {
            "total_cost": float(reward_sums.get("total_cost", 0.0)),
            "total_reward": float(reward_sums.get("reward", 0.0)),
            "onboard_passenger_delay_passenger_min": float(cumulative["passenger_delay"]),
            "average_excess_dwell_time_min": float(avg_excess),
            "total_bus_operating_delay_min": float(cumulative["bus_operating_delay"]),
            "parcel_lateness_parcel_min": float(cumulative["parcel_lateness"] + self.episode_metrics.get("terminal_penalty", 0.0)),
            "late_delivery_count": float(cumulative["late_delivery_count"]),
            "undelivered_parcel_count": float(len(self._terminal_undelivered_parcels(float(self.state.get("time", 0.0))))),
            "terminal_undelivered_penalty": float(self.episode_metrics.get("terminal_penalty", 0.0)),
            "minimum_bus_battery": float(min((b.get("battery_kwh", 0.0) for b in self.bus_states.values()), default=0.0)),
            "battery_safety_violation_count": float(sum(1 for b in self.bus_states.values() if float(b.get("battery_kwh", 0.0)) < float(b.get("safety_battery_kwh", 0.0)))),
            "total_energy_consumption": float(cumulative["energy_consumption"]),
            "station_power_overload_amount": float(cumulative["power_overload"]),
            "locker_overflow_amount": float(cumulative["locker_overflow"]),
            "station_power_overload_duration": float(cumulative["power_overload_duration"]),
            "locker_overflow_duration": float(cumulative["locker_overflow_duration"]),
            "average_charging_duration_min": float(self.episode_metrics.get("executed_duration_min_sum", 0.0) / decision_events) if decision_events else 0.0,
            "average_positive_charging_duration_min": (float(self.episode_metrics.get("positive_executed_duration_min_sum", 0.0)) / float(self.episode_metrics.get("charging_event_count", 0.0))) if float(self.episode_metrics.get("charging_event_count", 0.0)) else 0.0,
            "total_charging_duration_min": float(self.episode_metrics.get("executed_duration_min_sum", 0.0)),
            "mean_requested_charging_duration_min": float(self.episode_metrics.get("requested_duration_min_sum", 0.0) / decision_events) if decision_events else 0.0,
            "mean_executed_charging_duration_min": float(self.episode_metrics.get("executed_duration_min_sum", 0.0) / decision_events) if decision_events else 0.0,
            "mean_requested_action_index": float(self.episode_metrics.get("requested_action_sum", 0.0) / decision_events) if decision_events else 0.0,
            "mean_executed_action_index": float(self.episode_metrics.get("executed_action_sum", 0.0) / decision_events) if decision_events else 0.0,
            "charging_event_count": float(self.episode_metrics.get("charging_event_count", 0.0)),
            "zero_charging_event_count": float(self.episode_metrics.get("zero_charging_event_count", 0.0)),
            "valid_charging_opportunity_count": float(decision_events),
            "invalid_action_count": float(self.episode_metrics.get("invalid_action_count", 0.0)),
            "episode_length_decisions": float(decision_events),
            "reward_component_sums": reward_sums,
        }
        metrics.update({
            "onboard_passenger_delay": metrics["onboard_passenger_delay_passenger_min"],
            "average_excess_dwell_time": metrics["average_excess_dwell_time_min"],
            "total_bus_operating_delay": metrics["total_bus_operating_delay_min"],
            "parcel_lateness": metrics["parcel_lateness_parcel_min"],
            "average_charging_duration": metrics["average_charging_duration_min"],
            "mean_requested_action": metrics["mean_requested_action_index"],
            "mean_executed_action": metrics["mean_executed_action_index"],
        })
        return metrics

    def _refresh_global_state_features(self):
        now = float(self.state.get("time", 0.0))
        self.state["trip_progress"] = {tid: float(self.bus_states[tid]["current_stop_index"]) for tid in self.trip_ids}
        self.state["trip_battery"] = {tid: float(self.bus_states[tid]["battery_kwh"]) for tid in self.trip_ids}
        self.state["trip_onboard_pax"] = {tid: float(self.bus_states[tid]["onboard_passengers"]) for tid in self.trip_ids}
        self.state["trip_onboard_parcels"] = {tid: float(len(self.bus_states[tid]["onboard_parcel_ids"])) for tid in self.trip_ids}
        self.state["stop_queues"] = {sid: float(self.stop_queues.get(sid, 0)) for sid in self.stop_ids}
        for key in ["station_locker", "station_idle_drones", "station_full_batteries", "station_depleted_batteries", "station_charging_batteries", "station_power_consumption", "station_power_margin", "station_available_chargers", "station_earliest_charger_release", "station_earliest_drone_return", "station_earliest_battery_completion", "station_urg_min_slack", "station_urg_avg_slack", "station_urg_late_count", "station_urg_risk_count"]:
            self.state[key] = {}
        for sid, st in self.station_states.items():
            self.state["station_locker"][sid] = float(st.get("locker_inventory_kg", 0.0))
            self.state["station_idle_drones"][sid] = float(sum(1 for d in st.get("drones", []) if d.get("status") == "idle"))
            self.state["station_full_batteries"][sid] = float(st.get("full_batteries", 0))
            self.state["station_depleted_batteries"][sid] = float(st.get("depleted_batteries", 0))
            self.state["station_charging_batteries"][sid] = float(len(st.get("charging_batteries", [])))
            p_tot = float(st.get("current_bus_charging_load", 0.0)) + float(st.get("current_drone_battery_charging_load", 0.0)) + self._base_load_kw(st, now)
            self.state["station_power_consumption"][sid] = p_tot
            self.state["station_power_margin"][sid] = float(st["power_capacity_kw"]) - p_tot
            self.state["station_available_chargers"][sid] = float(self._available_chargers(st, now))
            self.state["station_earliest_charger_release"][sid] = max(0.0, min(st.get("charger_release_times_min", [now])) - now)
            active_missions = [m for m in st.get("active_drone_missions", []) if m.get("eta_return_min") is not None]
            self.state["station_earliest_drone_return"][sid] = max(0.0, min([m["eta_return_min"] for m in active_missions], default=now) - now)
            battery_jobs = st.get("charging_batteries", [])
            comp_times = [float(j.get("completion_time_min", now)) for j in battery_jobs if isinstance(j, dict)]
            self.state["station_earliest_battery_completion"][sid] = max(0.0, min(comp_times, default=now) - now)
            waits = [float(p["delivery_deadline_min"]) - now for p in self.parcel_states.values() if int(p.get("assigned_station_id", -999)) == sid and p.get("status") in {"onboard", "locker"}]
            self.state["station_urg_min_slack"][sid] = min(waits) if waits else self.horizon
            self.state["station_urg_avg_slack"][sid] = (sum(waits) / len(waits)) if waits else self.horizon
            self.state["station_urg_late_count"][sid] = float(sum(1 for w in waits if w < 0))
            self.state["station_urg_risk_count"][sid] = float(sum(1 for w in waits if w <= 15.0))
        arr_times = [float(b["next_arrival_time_min"]) for b in self.bus_states.values() if b.get("active", False)]
        self.state["future_arrivals_5m"] = sum(1 for t in arr_times if now <= t <= now + 5.0)
        self.state["future_arrivals_10m"] = sum(1 for t in arr_times if now <= t <= now + 10.0)
        self.state["future_arrivals_15m"] = sum(1 for t in arr_times if now <= t <= now + 15.0)

    def _build_obs_for_current_event(self):
        self._refresh_global_state_features()
        e = self.current_decision_event
        if e is None:
            local = {"station_id": -1, "trip_id": self.state.get("trip_id", -1), "arriving_battery": self.state.get("battery", 0.0), "onboard_before_alight": self.state.get("onboard_passengers", 0), "onboard_parcels_before_unload": self.state.get("onboard_parcels", 0), "alighting": 0, "initial_board": 0, "q_f": 0.0, "available_chargers": 0, "locker": self.state.get("locker", 0), "idle_drones": self.state.get("idle_drones", 0), "full_batteries": self.state.get("full_batteries", 0), "power_margin": self.state.get("power_margin", 0.0), "urg_min_slack": self.horizon, "urg_avg_slack": self.horizon, "urg_late_count": 0.0, "urg_risk_count": 0.0}
            return build_observation(self.state, local, self.obs_schema)
        bus = self._bus_for_trip(int(e.trip_id)); st = self.station_states[int(e.station_id)]
        unload_ids = get_unloading_parcels(bus["trip_id"], st["station_id"], self.assignment_index, self.parcel_states)
        preview = getattr(e, "passenger_service_preview", {}) or {}
        slacks = [float(p["delivery_deadline_min"]) - float(self.state["time"]) for p in self.parcel_states.values() if int(p.get("assigned_station_id", -1)) == int(st["station_id"]) and p.get("status") in {"onboard", "locker"}]
        local = {"station_id": st["station_id"], "trip_id": bus["trip_id"], "arriving_battery": bus["battery_kwh"], "onboard_before_alight": int(preview.get("onboard_before_alight", bus["onboard_passengers"])), "onboard_parcels_before_unload": len(bus["onboard_parcel_ids"]), "alighting": int(preview.get("alighting", 0)), "initial_board": int(preview.get("initial_board", 0)), "q_f": float(getattr(e, "unloading_volume_kg", compute_unloading_volume(unload_ids, self.parcel_states))), "available_chargers": self._available_chargers(st, self.state["time"]), "locker": st["locker_inventory_kg"], "idle_drones": sum(1 for d in st.get("drones", []) if d.get("status") == "idle"), "full_batteries": st.get("full_batteries", 0), "power_margin": st["power_capacity_kw"] - st.get("current_bus_charging_load", 0.0), "urg_min_slack": min(slacks) if slacks else self.horizon, "urg_avg_slack": (sum(slacks) / len(slacks)) if slacks else self.horizon, "urg_late_count": float(sum(1 for w in slacks if w < 0)), "urg_risk_count": float(sum(1 for w in slacks if w <= 15.0))}
        return build_observation(self.state, local, self.obs_schema)
