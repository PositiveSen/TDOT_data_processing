# TDOT Pavement Marking Item Grouping Analysis

## GROUP 1: Raised Pavement Markers - Snowplowable
**Description:** Snowplowable reflective markers with variations in directionality and color count
**Items:**
• 716-01.10: SNOWPLOWABLE REFLECTIVE MARKER
• 716-01.21: SNOWPLOWABLE RAISED PAVEMENT MARKERS (BI-DIR) (1 COLOR)
• 716-01.23: SNOWPLOWABLE RAISED PAVEMENT MARKERS (BI-DIR)(2 COLOR)
• 716-01.22: SNOWPLOWABLE RAISED PAVMENT MARKERS (MONO-DIR)(1 COLOR)
• 716-01.21: SNWPLWBLE PAVEMENT MRKRS (BI-DIR)(1 COLOR)
• 716-01.22: SNWPLWBLE PAVEMENT MRKRS (MONO-DIR)(1 COLOR)

**Consistent parsing approach:** 
- type="SNOWPLOWABLE_MARKER"
- directionality=["BI-DIR"|"MONO-DIR"|null]
- color_count=["1"|"2"|null]

---

## GROUP 2: Raised Pavement Markers - Standard
**Description:** Non-snowplowable raised markers with variations in directionality and color
**Items:**
• 716-01.12: RAISED PAVEMENT MARKERS (MONO-DIRECTIONAL) (1 COLOR LENS)
• 716-01.11: RAISED PAVEMENT MARKERS (BI-DIRECTIONAL) (1 COLOR LENS)
• 716-01.13: RAISED PAVEMENT MARKERS (BI-DIRECTIONAL) (2 COLOR LENS)
• 716-01.99: ROUND RAISED PAVEMENT MARKER

**Consistent parsing approach:**
- type="RAISED_MARKER"
- directionality=["BI-DIR"|"MONO-DIR"|"ROUND"|null]
- color_count=["1"|"2"|null]

---

## GROUP 3: Temporary Raised Pavement Markers
**Description:** Temporary markers with color variations
**Items:**
• 716-01.05: TEMPORARY RAISED PAVEMENT MARKER
• 716-01.06: TEMPORARY RAISED PAVEMENT MARKER,WHITE
• 716-01.07: TEMPORARY RAISED PAVEMENT MARKER,YELLOW

**Consistent parsing approach:**
- type="TEMPORARY_MARKER"
- color=["WHITE"|"YELLOW"|null]

---

## GROUP 4: Marker Maintenance/Removal
**Description:** Operations related to marker removal and lens replacement
**Items:**
• 716-01.30: REMOVAL OF SNOWPLOWABLE REFLECTIVE MARKER
• 716-01.40: REMOVE AND REPLACE LENS ON SNOWPLOWABLE REFLECTIVE MARKER
• 716-01.14: RAISED PAVEMENT MARKER REMOVAL
• 716-10.40: REMV & REPLC LENS ON SNOWPLW RFLCTV MKR

**Consistent parsing approach:**
- type="MARKER_MAINTENANCE"
- operation=["REMOVAL"|"LENS_REPLACEMENT"]
- marker_type=["SNOWPLOWABLE"|"RAISED"|null]

---

## GROUP 5: Painted Pavement Marking - Lines (Standard Width)
**Description:** Painted lines with standard widths (4", 6", 8", 12")
**Items:**
• 716-05.01: PAINTED PAVEMENT MARKING (4 INCH LINE)
• 716-05.20: PAINTED PAVEMENT MARKING (6 INCH LINE)
• 716-05.49: PAINTED PAVEMENT MARKINGS(8 INCH LINE)
• 716-05.50: PAINTED PAVEMENT MARKINGS(8 INCH LINE)
• 716-05.51: PAINTED PAVEMENT MARKINGS(12 INCH LINE)

**Consistent parsing approach:**
- material="PAINTED"
- type="LINE"
- width=["4"|"6"|"8"|"12"]
- unit="INCH"

---

## GROUP 6: Painted Pavement Marking - Barrier Lines
**Description:** Painted barrier lines with specific widths
**Items:**
• 716-05.02: PAINTED PAVEMENT MARKING (8 INCH BARRIER LINE)
• 716-05.07: PAINTED PAVEMENT MARKING (24 INCH BARRIER LINE)

**Consistent parsing approach:**
- material="PAINTED"
- type="BARRIER_LINE"
- width=["8"|"24"]
- unit="INCH"

---

## GROUP 7: Painted Pavement Marking - Dotted Lines
**Description:** Painted dotted lines with width variations
**Items:**
• 716-05.21: PAINTED PAVEMENT MARKING(4 INCH DOTTED LINE)
• 716-05.77: PERFORMANCE BASED RETRACING PAINTED 6 INCH DOTTED LINE
• 716-05.82: PERFORMANCE BASED RETRACING PAINTED 8 INCH DOTTED

**Consistent parsing approach:**
- material="PAINTED"
- type="DOTTED_LINE"
- width=["4"|"6"|"8"]
- unit="INCH"
- performance_based=[true|false]

---

## GROUP 8: Painted Pavement Marking - Symbols/Arrows
**Description:** Painted arrows and symbols
**Items:**
• 716-05.06: PAINTED PAVEMENT MARKING (TURN LANE ARROW)
• 716-05.09: PAINTED PAVEMENT MARKING(STRAIGHT-TURN ARROW)
• 716-05.11: PAINTED PAVEMENT MARKING(STRAIGHT ARROW)

**Consistent parsing approach:**
- material="PAINTED"
- type="ARROW"
- arrow_type=["TURN_LANE"|"STRAIGHT_TURN"|"STRAIGHT"]

---

## GROUP 9: Painted Pavement Marking - Special Features
**Description:** Painted special markings (crosswalks, stop lines, etc.)
**Items:**
• 716-05.03: PAINTED PAVEMENT MARKING (CROSS-WALK)
• 716-05.05: PAINTED PAVEMENT MARKING (STOP LINE)
• 716-05.22: PAINTED PAVEMENT MARKING (LONGITUDINAL CROSS-WALK)
• 716-05.04: PAINTED PAVEMENT MARKING (CHANNELIZATION STRIPING)
• 716-05.08: PAINTED PAVEMENT MARKING (PARKING LINE)
• 716-05.52: PAINTED PAVEMENT MARKING(AREA)

**Consistent parsing approach:**
- material="PAINTED"
- type=["CROSSWALK"|"STOP_LINE"|"CHANNELIZATION"|"PARKING_LINE"|"AREA"]
- modifier=["LONGITUDINAL"|null]

---

## GROUP 10: Painted Pavement Marking - Words
**Description:** Painted word markings
**Items:**
• 716-05.10: PAINTED PAVEMENT MARKING (RXR)
• 716-06.01: PAINTED WORD PAVEMENT MARK ( )

**Consistent parsing approach:**
- material="PAINTED"
- type="WORD"
- word_text=["RXR"|custom]

---

## GROUP 11: Painted Pavement Marking - Contrast Lines
**Description:** Painted contrast markings
**Items:**
• 716-05.12: PAINTED PAVEMENT MARKING (CONTRAST 6 INCH LINE)

**Consistent parsing approach:**
- material="PAINTED"
- type="CONTRAST_LINE"
- width="6"
- unit="INCH"

---

## GROUP 12: Plastic Pavement Marking - Lines (Standard Width)
**Description:** Plastic lines with standard widths (4", 6", 8", 12", 100mm, 150mm, 200mm)
**Items:**
• 716-02.12: PLASTIC PAVEMENT MARKING (8 INCH LINE)
• 716-02.01: PLASTIC PAVEMENT MARKING (100mm LINE)
• 716-02.10: PLASTIC PAVEMENT MARKING (150mm LINE)

**Consistent parsing approach:**
- material="PLASTIC"
- type="LINE"
- width=["4"|"6"|"8"|"12"|"100"|"150"|"200"]
- unit=["INCH"|"MM"]

---

## GROUP 13: Plastic Pavement Marking - Barrier Lines
**Description:** Plastic barrier lines with width variations
**Items:**
• 716-02.07: PLASTIC PAVEMENT MARKING (24 INCH BARRIER LINE)
• 716-02.23: PLASTIC PAVEMENT MARKING (12 INCH BARRIER LINE)
• 716-02.02: PLASTIC PAVEMENT MARKING (200mm BARRIER LINE)

**Consistent parsing approach:**
- material="PLASTIC"
- type="BARRIER_LINE"
- width=["12"|"24"|"200"]
- unit=["INCH"|"MM"]

---

## GROUP 14: Plastic Pavement Marking - Dotted Lines
**Description:** Plastic dotted lines with width variations
**Items:**
• 716-02.08: PLASTIC PAVEMENT MARKING (8 INCH DOTTED LINE)
• 716-04.03: PLASTIC PAVEMENT MARKING (4 INCH DOTTED LINE)
• 716-02.11: PLASTIC PAVEMENT MARKING (6 INCH DOTTED LINE)
• 716-02.08: PLASTIC PAVEMENT MARKING (200mm DOTTED LINE)
• 716-02.11: PLASTIC PAVEMENT MARKING (150mm DOTTED LINE)
• 716-09.33: 6 INCH DOTTED LINE

**Consistent parsing approach:**
- material="PLASTIC"
- type="DOTTED_LINE"
- width=["4"|"6"|"8"|"150"|"200"]
- unit=["INCH"|"MM"]

---

## GROUP 15: Plastic Pavement Marking - Special Width Lines
**Description:** Plastic lines with special designations (DWL, etc.)
**Items:**
• 716-02.24: PLASTIC PAVEMENT MARKING (12 INCH DWL)

**Consistent parsing approach:**
- material="PLASTIC"
- type="DWL"
- width="12"
- unit="INCH"

---

## GROUP 16: Plastic Pavement Marking - Arrows (Directional)
**Description:** Plastic arrow markings for lane direction
**Items:**
• 716-02.06: PLASTIC PAVEMENT MARKING (TURN LANE ARROW)
• 716-04.01: PLASTIC PAVEMENT MARKING (STRAIGHT-TURN ARROW)
• 716-04.02: PLASTIC PAVEMENT MARKING(DOUBLE TURNING ARROW)
• 716-04.05: PLASTIC PAVEMENT MARKING (STRAIGHT ARROW)
• 716-04.07: PLASTIC PAVEMENT MARKING (EXIT ONLY ARROW)
• 716-04.06: PLASTIC PAVEMENT MARKING (WRONG WAY ARROW)
• 716-04.08: PLASTIC PAVEMENT MARKING (OPTION LANE ARROW)
• 716-04.14: PLASTIC PAVEMENT MARKING (LANE REDUCTION ARROW)
• 716-02.15: PLASTIC PAVEMENT MARKING (U TURN ARROW)
• 716-09.32: EXIT ONLY ARROW

**Consistent parsing approach:**
- material="PLASTIC"
- type="ARROW"
- arrow_type=["TURN_LANE"|"STRAIGHT_TURN"|"DOUBLE_TURN"|"STRAIGHT"|"EXIT_ONLY"|"WRONG_WAY"|"OPTION_LANE"|"LANE_REDUCTION"|"U_TURN"]

---

## GROUP 17: Plastic Pavement Marking - Fish-Hook Arrows
**Description:** Plastic fish-hook arrow markings with varying arrow counts
**Items:**
• 716-04.23: PLASTIC PAVEMENT MARKING (FISH-HOOKS WITH 1 ARROW)
• 716-04.24: PLASTIC PAVEMENT MARKING (FISH-HOOKS WITH 2 ARROWS)
• 716-04.25: PLASTIC PAVEMENT MARKING (FISH-HOOKS WITH 3 ARROWS)

**Consistent parsing approach:**
- material="PLASTIC"
- type="FISH_HOOK_ARROW"
- arrow_count=["1"|"2"|"3"]

---

## GROUP 18: Plastic Pavement Marking - Crosswalks
**Description:** Plastic crosswalk markings with variations
**Items:**
• 716-02.03: PLASTIC PAVEMENT MARKING (CROSS-WALK)
• 716-02.09: PLASTIC PAVEMENT MARKING (LONGITUDINAL CROSS-WALK)
• 716-02.13: PLASTIC PAVEMENT MARKING (CROSSWALK)

**Consistent parsing approach:**
- material="PLASTIC"
- type="CROSSWALK"
- orientation=["STANDARD"|"LONGITUDINAL"]

---

## GROUP 19: Plastic Pavement Marking - Stop/Yield Lines
**Description:** Plastic stop and yield line markings
**Items:**
• 716-02.05: PLASTIC PAVEMENT MARKING (STOP LINE)
• 716-04.12: PLASTIC PAVEMENT MARKING (YIELD LINE)

**Consistent parsing approach:**
- material="PLASTIC"
- type=["STOP_LINE"|"YIELD_LINE"]

---

## GROUP 20: Plastic Pavement Marking - Symbols (Traffic Control)
**Description:** Plastic symbols for traffic control and information
**Items:**
• 716-04.09: PLASTIC PAVEMENT MARKING (H.O.V. DIAMOND)
• 716-04.10: PLASTIC PAVEMENT MARKING (HANDICAP SYMBOL)
• 716-04.17: PLASTIC PAVEMENT MARKING (YIELD SYMBOL)

**Consistent parsing approach:**
- material="PLASTIC"
- type="SYMBOL"
- symbol_type=["HOV_DIAMOND"|"HANDICAP"|"YIELD"]

---

## GROUP 21: Plastic Pavement Marking - Bike Symbols
**Description:** Plastic bicycle-related symbols and markings
**Items:**
• 716-04.15: PLASTIC PAVEMENT MARKING-BIKE SYMBOL/ARROW SHARED
• 716-04.13: PLASTIC PAVEMENT MARKING (BIKELANE SYMBOL & ARROW)
• 716-04.11: PLASTIC PAVEMENT MARKING (BICYCLE SYMBOL WITHRIDER)
• 716-04.18: PLASTIC PAVEMENT MARKING (BIKE/XING)

**Consistent parsing approach:**
- material="PLASTIC"
- type="BIKE_SYMBOL"
- symbol_variant=["SHARED_ARROW"|"LANE_ARROW"|"WITH_RIDER"|"CROSSING"]

---

## GROUP 22: Plastic Pavement Marking - Special Features
**Description:** Plastic special purpose markings
**Items:**
• 716-02.04: PLASTIC PAVEMENT MARKING(CHANNELIZATION STRIPING)
• 716-04.04: PLASTIC PAVEMENT MARKING (TRANSVERSE SHOULDER)
• 716-04.16: PLASTIC PAVEMENT MARKING (NOISE STRIP)
• 716-02.22: PLASTIC AERIAL SPEED BARS

**Consistent parsing approach:**
- material="PLASTIC"
- type=["CHANNELIZATION"|"TRANSVERSE_SHOULDER"|"NOISE_STRIP"|"SPEED_BARS"]

---

## GROUP 23: Plastic Word Pavement Marking - Standard Words
**Description:** Plastic word markings with specific text
**Items:**
• 716-03.01: PLASTIC WORD PAVEMENT MARKING (ONLY)
• 716-03.02: PLASTIC WORD PAVEMENT MARKING (RXR)
• 716-03.03: PLASTIC WORD PAVEMENT MARKING (STOP AHEAD)
• 716-03.04: PLASTIC WORD PAVEMENT MARKING (SCHOOL)
• 716-03.05: PLASTIC WORD PAVEMENT MARKING (BIKE LANE)
• 716-03.06: PLASTIC WORD PAVEMENT MARKING (SIGNAL AHEAD)
• 716-03.07: PLASTIC WORD PAVEMENT MARKING (STOP)
• 716-03.08: PLASTIC WORD PAVEMENT MARKING (PED-XING)
• 716-03.13: PLASTIC PAVEMENT MARKING (NO TRUCKS THIS LANE)

**Consistent parsing approach:**
- material="PLASTIC"
- type="WORD"
- word_text=["ONLY"|"RXR"|"STOP_AHEAD"|"SCHOOL"|"BIKE_LANE"|"SIGNAL_AHEAD"|"STOP"|"PED_XING"|"NO_TRUCKS"]

---

## GROUP 24: Plastic Word Pavement Marking - Custom/Variable
**Description:** Plastic word markings with custom or variable text
**Items:**
• 716-03.09: PLASTIC WORD PAVEMENT MARKING ( )
• 716-03.10: PLASTIC WORD PAVEMENT MARKING ( )
• 716-03.11: PLASTIC WORD PAVEMENT MARKING ( )
• 716-03.12: PLASTIC PAVEMENT MARKING (DESCRIPTION)

**Consistent parsing approach:**
- material="PLASTIC"
- type="WORD"
- word_text="CUSTOM"
- custom_text=[from description]

---

## GROUP 25: Enhanced Flatline Thermoplastic - Lines (Standard Width)
**Description:** Enhanced flatline thermoplastic lines with standard widths
**Items:**
• 716-12.01: ENHANCED FLATLINE THERMOPLASTIC PAVEMENT MARKING (4 INCH LINE)
• 716-12.02: ENHANCED FLATLINE THERMOPLASTIC PAVEMENT MARKING (6 INCH LINE)
• 716-12.06: ENHANCED FLATLINE THERMOPLASTIC (8 INCH LINE)
• 716-12.09: ENHANCED FLATLINE THERMOPLASTIC (12 INCH LINE)
• 716-12.02: ENHNCD FLTLNE THERMOPLASTIC PAVEMENT MARKING (6 INCH LINE)

**Consistent parsing approach:**
- material="THERMOPLASTIC"
- subtype="ENHANCED_FLATLINE"
- type="LINE"
- width=["4"|"6"|"8"|"12"]
- unit="INCH"

---

## GROUP 26: Enhanced Flatline Thermoplastic - Barrier Lines
**Description:** Enhanced flatline thermoplastic barrier lines
**Items:**
• 716-12.03: ENHANCED FLATLINE THERMOPLASTIC PAVEMENT MARKING (8 INCH BARRIER LINE)
• 716-12.08: ENHANCED FLATLINE THERMOPLASTIC (12 INCH BARRIER LINE)

**Consistent parsing approach:**
- material="THERMOPLASTIC"
- subtype="ENHANCED_FLATLINE"
- type="BARRIER_LINE"
- width=["8"|"12"]
- unit="INCH"

---

## GROUP 27: Enhanced Flatline Thermoplastic - Dotted Lines
**Description:** Enhanced flatline thermoplastic dotted lines
**Items:**
• 716-12.04: ENHANCED FLATLINE THERMOPLASTIC PAVEMENT MARKING (4 INCH DOTTED LINE)
• 716-12.05: ENHANCED FLATLINE THERMOPLASTIC PAVEMENT MARKING (6 INCH DOTTED LINE)
• 716-12.10: ENHANCED FLATLINE THERMOPLASTIC (12 INCH DOTTED)

**Consistent parsing approach:**
- material="THERMOPLASTIC"
- subtype="ENHANCED_FLATLINE"
- type="DOTTED_LINE"
- width=["4"|"6"|"12"]
- unit="INCH"

---

## GROUP 28: Enhanced Flatline Thermoplastic - Broken Lines
**Description:** Enhanced flatline thermoplastic broken lines
**Items:**
• 716-12.07: ENHANCED FLATLINE THERMOPLASTIC (8 INCH BROKEN LN)

**Consistent parsing approach:**
- material="THERMOPLASTIC"
- subtype="ENHANCED_FLATLINE"
- type="BROKEN_LINE"
- width="8"
- unit="INCH"

---

## GROUP 29: Spray Thermoplastic - 40 mil Lines
**Description:** Spray thermoplastic 40 mil thickness lines with width variations
**Items:**
• 716-13.06: SPRAY THERMOPLASTIC PAVEMENT MARKING (40 mil) (4 INCH LINE)
• 716-13.07: SPRAY THERMOPLASTIC PAVEMENT MARKING (40 mil) (6 INCH LINE)
• 716-13.08: SPRAY THERMOPLASTIC PAVEMENT MARKING (40 mil) (8 INCH BARRIER LINE)

**Consistent parsing approach:**
- material="THERMOPLASTIC"
- subtype="SPRAY"
- thickness="40"
- thickness_unit="MIL"
- type=["LINE"|"BARRIER_LINE"]
- width=["4"|"6"|"8"]
- unit="INCH"

---

## GROUP 30: Spray Thermoplastic - 40 mil Dotted Lines
**Description:** Spray thermoplastic 40 mil thickness dotted lines
**Items:**
• 716-13.09: SPRAY THERMOPLASTIC PAVEMENT MARKING (40 mil) (4 INCH DOTTED LINE)
• 716-13.10: SPRAY THERMOPLASTIC PAVEMENT MARKING (40 mil) (6 INCH DOTTED LINE)

**Consistent parsing approach:**
- material="THERMOPLASTIC"
- subtype="SPRAY"
- thickness="40"
- thickness_unit="MIL"
- type="DOTTED_LINE"
- width=["4"|"6"]
- unit="INCH"

---

## GROUP 31: Spray Thermoplastic - 60 mil Lines
**Description:** Spray thermoplastic 60 mil thickness lines with width variations
**Items:**
• 716-13.01: SPRAY THERMOPLASTIC PAVEMENT MARKING (60 mil) (4 INCH LINE)
• 716-13.02: SPRAY THERMOPLASTIC PAVEMENT MARKING (60 mil) (6 INCH LINE)
• 716-13.03: SPRAY THERMOPLASTIC PAVEMENT MARKING (60 mil) (8 INCH BARRIER LINE)
• 716-13.11: SPRAY THERMOPLASTIC PAVEMENT MARKING (60 MIL 12 INCH BARRIER LINE)
• 716-13.01: SPRAY THERMOPLASTIC PAVEMENT MARKING (60 mil) (100 mm)
• 716-13.02: SPRAY THERMOPLASTIC PAVEMENT MARKING (60 mil) (200mm LINE)

**Consistent parsing approach:**
- material="THERMOPLASTIC"
- subtype="SPRAY"
- thickness="60"
- thickness_unit="MIL"
- type=["LINE"|"BARRIER_LINE"]
- width=["4"|"6"|"8"|"12"|"100"|"200"]
- unit=["INCH"|"MM"]

---

## GROUP 32: Spray Thermoplastic - 60 mil Dotted Lines
**Description:** Spray thermoplastic 60 mil thickness dotted lines
**Items:**
• 716-13.04: SPRAY THERMOPLASTIC PAVEMENT MARKING (60 mil) (4 INCH DOTTED LINE)
• 716-13.05: SPRAY THERMOPLASTIC PAVEMENT MARKING (60 mil) (6 INCH DOTTED LINE)

**Consistent parsing approach:**
- material="THERMOPLASTIC"
- subtype="SPRAY"
- thickness="60"
- thickness_unit="MIL"
- type="DOTTED_LINE"
- width=["4"|"6"]
- unit="INCH"

---

## GROUP 33: Spray Thermoplastic - Special Lines
**Description:** Spray thermoplastic special purpose lines
**Items:**
• 716-13.12: SPRAY THERMOPLASTIC PAVEMENT MARKING (60 mil) (6 INCH BLACK LEAD-LAG LINE)
• 716-11.01: SPRAY THERMOPLASTIC PAVEMENT MARKING (4 INCH LINE)

**Consistent parsing approach:**
- material="THERMOPLASTIC"
- subtype="SPRAY"
- thickness=["60"|null]
- thickness_unit="MIL"
- type=["LEAD_LAG_LINE"|"LINE"]
- width=["4"|"6"]
- unit="INCH"
- color=["BLACK"|null]

---

## GROUP 34: Profiled Thermoplastic - Audible Lines
**Description:** Profiled thermoplastic with audible features
**Items:**
• 716-14.01: PROFILED THERMOPLASTIC PAVEMENT MARKING AUDIBLE (4 INCH)
• 716-14.02: PROFILED THERMOPLASTIC PAVEMENT MARKING AUDIBLE (6 INCH)
• 716-09.79: THERMOPLST PAVEMENT MARK PROFILE LINE(4 INCH )
• 716-09.80: THERMOPLST PAVEMENT MARK PROFILE LINE(6 INCH )

**Consistent parsing approach:**
- material="THERMOPLASTIC"
- subtype="PROFILED"
- feature="AUDIBLE"
- type="LINE"
- width=["4"|"6"]
- unit="INCH"

---

## GROUP 35: Performance Based Retracing - Spray Thermoplastic Lines
**Description:** Performance-based retracing spray thermoplastic lines
**Items:**
• 716-05.72: PERFORMANCE BASED RETRACING SPRAY THERMOPLASTIC 4 INCH
• 716-05.73: PERFORMANCE BASED RETRACING SPRAY THERMOPLASTIC 6 INCH
• 716-05.74: PERFORMANCE BASED RETRACING SPRAY THERMOPLASTIC 8 INCH
• 716-05.79: PERFORMANCE BASED RETRACING SPRAY THERMOPLASTIC 12 INCH

**Consistent parsing approach:**
- material="THERMOPLASTIC"
- subtype="SPRAY"
- operation="PERFORMANCE_RETRACING"
- type="LINE"
- width=["4"|"6"|"8"|"12"]
- unit="INCH"

---

## GROUP 36: Performance Based Retracing - Spray Thermoplastic Dotted Lines
**Description:** Performance-based retracing spray thermoplastic dotted lines
**Items:**
• 716-05.75: PERFORMANCE BASED RETRACING SPRAY THERMOPLASTIC 6 INCH DOTTED
• 716-05.80: PERFORMANCE BASED RETRACING SPRAY THERMOPLASTIC 8 INCH DOTTED
• 716-05.81: PERFORMANCE BASED RETRACING SPRAY THERMOPLASTIC 12 INCH DOTTED

**Consistent parsing approach:**
- material="THERMOPLASTIC"
- subtype="SPRAY"
- operation="PERFORMANCE_RETRACING"
- type="DOTTED_LINE"
- width=["6"|"8"|"12"]
- unit="INCH"

---

## GROUP 37: Performance Based Retracing - Painted Lines
**Description:** Performance-based retracing painted lines
**Items:**
• 716-05.69: PERFORMANCE BASED RETRACING PAINTED 4 INCH LINE
• 716-05.70: PERFORMANCE BASED RETRACING PAINTED 6 INCH LINE
• 716-05.71: PERFORMANCE BASED RETRACING PAINTED 8 INCH LINE

**Consistent parsing approach:**
- material="PAINTED"
- operation="PERFORMANCE_RETRACING"
- type="LINE"
- width=["4"|"6"|"8"]
- unit="INCH"

---

## GROUP 38: Retracing Pavement Markings - Plastic
**Description:** Retracing operations for plastic markings
**Items:**
• 716-02.31: RETRACING PAVEMENT MARKINGS - PLASTIC (8 INCH BARRIER LINE)
• 716-02.32: RETRACING PAVEMENT MARKINGS - PLASTIC (6 INCH LINE)
• 716-05.64: RETRACING PAVEMENT MARKINGS - PAINTED(6 INCH DOTTED LINE)

**Consistent parsing approach:**
- material=["PLASTIC"|"PAINTED"]
- operation="RETRACING"
- type=["BARRIER_LINE"|"LINE"|"DOTTED_LINE"]
- width=["6"|"8"]
- unit="INCH"

---

## GROUP 39: Wet Reflective Pavement Marking - Lines
**Description:** Wet reflective pavement markings with width variations
**Items:**
• 716-09.72: WET REFLECTIVE PAVEMEMT MARKING (4 INCH LINE)
• 716-09.03: WET REFLECTIVE PAVEMENT MARKING(6 INCH LINE)
• 716-09.02: WET REFLEC. PAVEMENT MARKING(8 INCH BARRIER LINE)
• 716-10.26: WET NIGHT VISIBLE PAVEMENT MARKING (6 INCH DOTTED LINE)

**Consistent parsing approach:**
- material="WET_REFLECTIVE"
- type=["LINE"|"BARRIER_LINE"|"DOTTED_LINE"]
- width=["4"|"6"|"8"]
- unit="INCH"
- feature=["NIGHT_VISIBLE"|null]

---

## GROUP 40: Contrast Pavement Marking - Lines
**Description:** Contrast pavement markings with width variations
**Items:**
• 716-09.85: CONTRAST PAVEMENT MARKING 4 INCH
• 716-09.86: CONTRAST PAVEMENT MARKING 6 INCH
• 716-09.83: CONTRAST PAVEMENT MARKING 6 INCH
• 716-09.84: CONTRAST PAVEMENT MARKING 8 INCH
• 716-09.88: CONTRAST PAVEMENT MARKING 8 INCH
• 716-09.89: CONTRAST PAVEMENT MARKING 12 INCH

**Consistent parsing approach:**
- material="CONTRAST"
- type="LINE"
- width=["4"|"6"|"8"|"12"]
- unit="INCH"

---

## GROUP 41: Contrast Pavement Marking - Dotted Lines
**Description:** Contrast pavement dotted line markings
**Items:**
• 716-09.90: CONTRAST PAVEMENT MARKING 6 INCH DOTTED

**Consistent parsing approach:**
- material="CONTRAST"
- type="DOTTED_LINE"
- width="6"
- unit="INCH"

---

## GROUP 42: Contrast Pavement Marking - Shadow Lines
**Description:** Contrast shadow pavement markings
**Items:**
• 716-09.94: CONTRAST PAVEMENT SHADOW MARKING 6 INCH
• 716-09.95: CONTRAST PAVEMENT SHADOW MARKING 8 INCH
• 716-09.97: CONTRAST PAVEMENT SHADOW MARKING 6 INCH
• 716-09.98: CONTRAST PAVEMENT SHADOW MARKING 8 INCH
• 716-09.57: CONTRAST PAVEMENT SHADOW MARKING 6 INCH (TAPE)

**Consistent parsing approach:**
- material="CONTRAST"
- subtype="SHADOW"
- type="LINE"
- width=["6"|"8"]
- unit="INCH"
- application=["TAPE"|null]

---

## GROUP 43: Contrast Pavement Marking - Words/Symbols
**Description:** Contrast markings for words and symbols
**Items:**
• 716-09.87: CONTRAST PAVEMENT MARKINGS WORDS AND SYMBOLS

**Consistent parsing approach:**
- material="CONTRAST"
- type="WORD_SYMBOL"

---

## GROUP 44: Preformed Plastic Pavement Marking - Lines
**Description:** Preformed plastic markings with contrast features
**Items:**
• 716-10.23: PREFORMED PLSTC PVMNT MRKING 4 INCH LINE WITH1.5 INCH BLACK SILHOUETTE

**Consistent parsing approach:**
- material="PREFORMED_PLASTIC"
- type="LINE"
- width="4"
- unit="INCH"
- contrast_width="1.5"
- contrast_color="BLACK"

---

## GROUP 45: Preformed Plastic Pavement Marking - Special Features
**Description:** Preformed plastic special markings
**Items:**
• 716-10.07: PREFORMED PLASTIC PAVEMENT MARKING (STOP LINE)
• 716-10.50: PREFORMED PLASTIC PAVEMENT MARKING (INTERSTATE SHIELD)
• 716-10.51: PRFRMED PLSTC PVMNT MARKING (STATE SHLD)
• 716-10.52: PRFRMED PLSTC PVMNT MARKING (US SHLD)

**Consistent parsing approach:**
- material="PREFORMED_PLASTIC"
- type=["STOP_LINE"|"SHIELD"]
- shield_type=["INTERSTATE"|"STATE"|"US"|null]

---

## GROUP 46: Preformed Plastic Pavement Marking - Words
**Description:** Preformed plastic word markings
**Items:**
• 716-10.15: PREFORMED PLASTIC PAVEMENT MARKING (NO TRUCKS THIS LANE)
• 716-16.10: PREFORMED YIELD WORD (8FT W/ 1 ½IN BLACK CONTRAST)

**Consistent parsing approach:**
- material="PREFORMED_PLASTIC"
- type="WORD"
- word_text=["NO_TRUCKS"|"YIELD"]
- contrast=[true|false]

---

## GROUP 47: Preformed Permanent Tape - Lines
**Description:** Preformed permanent tape with width variations
**Items:**
• 716-15.10: PREFORMED PERMANENT TAPE (6 INCH LINE)
• 716-15.11: PREFORMED PERMANENT TAPE (8 INCH LINE)
• 716-15.12: PREFORMED PERMANENT TAPE (12 INCH LINE)
• 716-15.20: PREFORMED PERMANENT TAPE (24 INCH BARRIER LINE)

**Consistent parsing approach:**
- material="PREFORMED_TAPE"
- type=["LINE"|"BARRIER_LINE"]
- width=["6"|"8"|"12"|"24"]
- unit="INCH"

---

## GROUP 48: Polyurea Pavement Marking
**Description:** Polyurea pavement markings with various types
**Items:**
• 716-40.01: POLYUREA PAVEMENT MARKING (6 INCH LINE)
• 716-40.02: POLYUREA PAVEMENT MARKINGS-WORDS-SYMBOLS
• 716-40.03: POLYUREA PAVEMENT MARKING (STOP BAR)

**Consistent parsing approach:**
- material="POLYUREA"
- type=["LINE"|"WORD_SYMBOL"|"STOP_BAR"]
- width=["6"|null]
- unit=["INCH"|null]

---

## GROUP 49: Textured Buffer/Lane Markings
**Description:** Textured buffer and lane markings with color variations
**Items:**
• 716-15.01: YELLOW TEXTURED BUFFER
• 716-04.21: GREEN TEXTURED BIKE LANE

**Consistent parsing approach:**
- material="TEXTURED"
- type=["BUFFER"|"BIKE_LANE"]
- color=["YELLOW"|"GREEN"]

---

## GROUP 50: Special Pavement Features
**Description:** Special pavement features and accessories
**Items:**
• 716-10.30: TRUNCATED DOME DETECTABLE WARNING MAT
• 716-80.01: PROTECTIVE BOLLARDS

**Consistent parsing approach:**
- type=["WARNING_MAT"|"BOLLARD"]
- subtype=["TRUNCATED_DOME"|"PROTECTIVE"]

---

## GROUP 51: Grooving for Pavement Marking
**Description:** Grooving operations for recessed markings
**Items:**
• 716-40.42: GROOVING FOR RECESSED PAVEMENT MARKING (SOLID)

**Consistent parsing approach:**
- operation="GROOVING"
- type="RECESSED_MARKING"
- pattern="SOLID"

---

## GROUP 52: Removal of Pavement Marking - Lines
**Description:** Removal operations for line markings
**Items:**
• 716-08.01: REMOVAL OF PAVEMENT MARKING (LINE)
• 716-08.20: REMOVAL OF PAVEMENT MARKING (LINE)
• 716-08.30: HYDROBLAST REMOVAL OF PAVEMENT MARKING (LINE)

**Consistent parsing approach:**
- operation="REMOVAL"
- type="LINE"
- method=["STANDARD"|"HYDROBLAST"]

---

## GROUP 53: Removal of Pavement Marking - Barrier Lines
**Description:** Removal operations for barrier line markings
**Items:**
• 716-08.02: REMOVAL OF PAVEMENT MARKING (8 INCH BARRIER LINE)
• 716-08.21: REMOVAL OF PAVEMENT MARKING (24 INCH BARRIER LINE)

**Consistent parsing approach:**
- operation="REMOVAL"
- type="BARRIER_LINE"
- width=["8"|"24"]
- unit="INCH"

---

## GROUP 54: Removal of Pavement Marking - Dotted Lines
**Description:** Removal operations for dotted line markings
**Items:**
• 716-08.09: REMOVAL OF PAVEMENT MARKING (DOTTED LINE)

**Consistent parsing approach:**
- operation="REMOVAL"
- type="DOTTED_LINE"

---

## GROUP 55: Removal of Pavement Marking - Special Features
**Description:** Removal operations for special markings
**Items:**
• 716-08.03: REMOVAL OF PAVEMENT MARKING (CROSS-WALK)
• 716-08.04: REMOVAL OF PAVEMENT MARKING (CHANNELIZATION STRIPING)
• 716-08.05: REMOVAL OF PAVEMENT MARKING (STOP LINE)
• 716-08.10: REMOVAL OF PAVEMENT MARKING (TRANSVERSE SHOULDER)
• 716-08.19: REMOVAL OF PAVEMENT MARKING (YIELD LINE)
• 716-08.22: REMOVAL OF PAVEMENT MARKING (NOISE STRIP)

**Consistent parsing approach:**
- operation="REMOVAL"
- type=["CROSSWALK"|"CHANNELIZATION"|"STOP_LINE"|"TRANSVERSE_SHOULDER"|"YIELD_LINE"|"NOISE_STRIP"]

---

## GROUP 56: Removal of Pavement Marking - Arrows
**Description:** Removal operations for arrow markings
**Items:**
• 716-08.06: REMOVAL OF PAVEMENT MARKING (TURN LANE ARROW)
• 716-08.07: REMOVAL OF PAVEMENT MARKING (STRAIGHT-TURN ARROW)
• 716-08.08: REMOVAL OF PAVEMENT MARKING (DOUBLE TURNING ARROW)
• 716-08.18: REMOVAL OF PAVEMENT MARKING (WRONG WAY ARROW)
• 716-08.23: REMOVAL OF PAVEMENT MARKING (STRAIGHT ARROW)
• 716-08.24: REMOVAL OF PAVEMENT MARKING (LANE REDUCTION ARROW)

**Consistent parsing approach:**
- operation="REMOVAL"
- type="ARROW"
- arrow_type=["TURN_LANE"|"STRAIGHT_TURN"|"DOUBLE_TURN"|"WRONG_WAY"|"STRAIGHT"|"LANE_REDUCTION"]

---

## GROUP 57: Roadway Preparation
**Description:** Roadway preparation for pavement marking
**Items:**
• 716-50.01: ROADWAY CLEANING FOR PAVEMENT MARKING

**Consistent parsing approach:**
- operation="PREPARATION"
- type="CLEANING"
- purpose="PAVEMENT_MARKING"

---

## SUMMARY OF PARSING BENEFITS:

This grouping approach ensures:

1. **Dimensional Consistency**: All width measurements parsed uniformly (4", 6", 8", etc.)
2. **Material Consistency**: Same material types grouped together (PAINTED, PLASTIC, THERMOPLASTIC, etc.)
3. **Type Consistency**: Similar marking types (LINE, DOTTED_LINE, BARRIER_LINE, ARROW, etc.) parsed consistently
4. **Operation Consistency**: Operations (REMOVAL, RETRACING, PERFORMANCE_RETRACING) handled uniformly
5. **Feature Extraction**: Consistent extraction of thickness, width, color, directionality, and other attributes
6. **Reduced Categories**: From 200+ individual items to ~57 logical groups with consistent parsing rules

This structure allows ML models to:
- Recognize patterns across similar items
- Predict costs based on material + type + dimensions
- Handle new items by matching to existing groups
- Reduce overfitting from treating similar items as completely different categories