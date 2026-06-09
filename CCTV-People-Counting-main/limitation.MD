## Known Limitations

### System Started Mid-Operation
The system was initialized during active working hours, not at the start of the day.
To account for people already inside the area at deployment time, `STARTING_COUNT`
in `config.py` was manually set to `30`.

This means the count does not represent total visitors from the beginning of the day,
but tracks net occupancy from the time the system was started.

**Recommendation:** For full-day tracking, start the system before operational hours
begin with `STARTING_COUNT = 0`.
