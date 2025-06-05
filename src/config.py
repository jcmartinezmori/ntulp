# --- system parameters --- #
RELPATH = '.'
BASIC = 0
CBASIC = 0

# --- instance parameters --- #
LINES_DIST_TRGT = 400  # meters
LINES_DIST_CTFF = 1600  # meters
TRIP_MILES_LB = 1/2  # miles
TRIP_SECS_LB = 60  # seconds
PLACE = 'Cook County'
FILENAME = PLACE.replace(' ', '-')
ADMIN_LEVEL = 6
CUSTOM_FILTER = (
    '['
    '"highway"~'
    '"motorway|trunk|primary|secondary|tertiary|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|residential"'
    ']'
)
STOPS_TAGS = {
    'highway': 'bus_stop'
}
AIRPORT_NODES = {310333647, 365195019}


# --- plotting parameters --- #
CENTER = (41.8781, -87.6298)
LINESCALING = 4
HEXBLACK = "#000000"
HEXORANGE = "#E69F00"
HEXSKYBLUE = "#56B4E9"
HEXBLUISHGREEN = "#009E73"
HEXYELLOW = "#F0E442"
HEXBLUE = "#0072B2"
HEXVERMILLION = "#D55E00"
HEXREDDISHPURPUPLE = "#CC79A7"
HEXCOLORS = [
    HEXBLUE,
    HEXYELLOW,
    HEXVERMILLION,
    HEXSKYBLUE,
    HEXORANGE,
    HEXBLUISHGREEN,
    HEXREDDISHPURPUPLE,
]
MARKERS = [
    'circle', 'square', 'diamond', 'cross', 'x', 'hexagram'
]
