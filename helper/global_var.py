# this is global vars

KEY_MULT_DICT = {
    "x-y-w-h": {"x":0, "y": 1, "w": 2, "h": 3},
    "xywh": {},
}


MULTI_CHOICE = {
    "kmeans": "x-y-w-h",
    "linear": "xywh",
    "int2str": "",
    "float2str": "",
}


SPECIAL_TOKENS = ["<MASK>", "<PAD>", "<EOS>", "<BOS>", "<SEP>"]

IGNORE_INDEX = -100

TEMPLATE_FORMAT = {
    "html_format": "<body> <svg width=\"{W}\" height=\"{H}\"> {content} </svg> </body>",
    "bbox_format": "<rect data-category=\"{c}\", x=\"{x}\", y=\"{y}\", width=\"{w}\", height=\"{h}\", file_name=\"{file_name}\"/>",
}

TASK_INSTRUCTION = {
    "rico25": "I want to generate layout in the mobile app design format. ",
    "publaynet": "I want to generate layout in the document design format. ",
    "magazine": "I want to generate layout in the magazine design format. ",
    "cgl" : "I want to generate layout in poster design format. ",
    "pku" : "I want to generate layout in poster design format. ",
    "miridih": "I want to generate layout in {template_type} design format. "
}
#INSTRUCTION = {
#    "cond_cate_to_size_pos": "please generate the layout html according to the categories I provide (in html format):\n###bbox html: {bbox_html}",
#    "cond_cate_size_to_pos": "please generate the layout html according to the categories and size I provide (in html format):\n###bbox html: {bbox_html}",
#    "cond_random_mask": "please recover the layout html according to the bbox, categories and size I provide (in html format):\n###bbox html: {bbox_html}"
#}
INSTRUCTION = {
    # c -> s,b
    "cond_cate_to_size_pos": "please generate the layout html according to the categories and image I provide (in html format):\n###bbox html: {bbox_html}",
    # c,s -> b
    "cond_cate_size_to_pos": "please generate the layout html according to the categories and size and image I provide (in html format):\n###bbox html: {bbox_html}",
    # c,b -> s
    "cond_cate_pos_to_size" : "please generate the layout html according to the categories and position and image I provide (in html format):\n###bbox html: {bbox_html}",#
    # recover
    "cond_random_mask": "please recover the layout html according to the bbox , categories, size, image I provide (in html format):\n###bbox html: {bbox_html}",
    # unconditional
    "unconditional" : "plaese generate the layout html according to the image I provide (in html format):\n###bbox html: {bbox_html}",#
    # refinement
    "refinement" : "please refine the layout html according to the image I provide (in html format):\n###bbox html: {bbox_html}",#
    # completion 
    "completion" : "please complete the layout html according to the image and element I provide (in html format):\n###bbox html: {bbox_html}",#
    

}

TEXT_INSTRUCTION = {
    # c -> s,b
    "cond_cate_to_size_pos": "please generate the layout html according to the categories and image I provide (in html format)\nText: {text}\n###bbox html: {bbox_html}",
    # c,s -> b
    "cond_cate_size_to_pos": "please generate the layout html according to the categories and size and image I provide (in html format)\nText: {text}\n###bbox html: {bbox_html}",
    # c,b -> s
    "cond_cate_pos_to_size" : "please generate the layout html according to the categories and position and image I provide (in html format)\nText: {text}\n###bbox html: {bbox_html}",#
    # recover
    "cond_random_mask": "please recover the layout html according to the bbox , categories, size, image I provide (in html format)\nText: {text}\n###bbox html: {bbox_html}",
    # unconditional
    "unconditional" : "plaese generate the layout html according to the image I provide (in html format)\nText: {text}\n###bbox html: {bbox_html}",#
    # refinement
    "refinement" : "please refine the layout html according to the image I provide (in html format)\nText: {text}\n###bbox html: {bbox_html}",#
    # completion 
    "completion" : "please complete the layout html according to the image and element I provide (in html format)\nText: {text}\n###bbox html: {bbox_html}",#
    

}


INFILLING_INSTRUCTION = {
    "cond_cate_to_size_pos": "please fulfilling the layout html according to the categories I provide (in html format):\n###bbox html: {bbox_html}",
    "cond_cate_size_to_pos": "please fulfilling the layout html according to the categories and size I provide (in html format):\n###bbox html: {bbox_html}",
    "cond_random_mask": "please recover the layout html according to the bbox, categories and size I provide (in html format):\n###bbox html: {bbox_html}"
}

SEP_SEQ = [
    "{instruct}\n\n##Here is the result:\n\n```{result}```",
    "{instruct}\n\n##Here is the result:",
    "{instruct} <MID> {result}",
    "{instruct} <MID>",
]

DATASET_META = {
    "magazine": {
        0: 'text',
        1: 'image',
        2: 'headline',
        3: 'text-over-image',
        4: 'headline-over-image',
    },
    "publaynet": {
        0: 'text',
        1: 'title',
        2: 'list',
        3: 'table',
        4: 'figure',
    },
    "rico25": {
        0: "Text",
        1: "Image",
        2: "Icon",
        3: "Text Button",
        4: "List Item",
        5: "Input",
        6: "Background Image",
        7: "Card",
        8: "Web View",
        9: "Radio Button",
        10: "Drawer",
        11: "Checkbox",
        12: "Advertisement",
        13: "Modal",
        14: "Pager Indicator",
        15: "Slider",
        16: "On/Off Switch",
        17: "Button Bar",
        18: "Toolbar",
        19: "Number Stepper",
        20: "Multi-Tab",
        21: "Date Picker",
        22: "Map View",
        23: "Video",
        24: "Bottom Navigation",
    },
    "cgl": {
        1: "Logo",
        2: "Text",
        3: "Underlay",
        4: "Embellishment",
        5: "Highlighted text"
        
    },
    "pku" :{
        1: "Text",
        2: "Logo",
        3: "Underlay"
    },
    "miridih":{
        1: "TEXT",
        2: "GENERALSVG",
        3: "SHAPESVG",
        4: "PHOTO",
        5: "FrameItem",
        6: "LineShapeItem",
        7: "GRID",
        8: "Chart",
        9: "GIF",
        10: "QRCode",
        11: "VIDEO",
        12: "Barcode",
        13: "YOUTUBE",
        14: "BASICSVG"
    }
}

# verborsed number
VERBALIZED_NUM = {  
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',  
    10: 'ten', 11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen', 15: 'fifteen', 16: 'sixteen',  
    17: 'seventeen', 18: 'eighteen', 19: 'nineteen', 20: 'twenty',  
    21: 'twenty-one', 22: 'twenty-two', 23: 'twenty-three', 24: 'twenty-four', 25: 'twenty-five',  
    26: 'twenty-six', 27: 'twenty-seven', 28: 'twenty-eight', 29: 'twenty-nine', 30: 'thirty',  
    31: 'thirty-one', 32: 'thirty-two', 33: 'thirty-three', 34: 'thirty-four', 35: 'thirty-five',  
    36: 'thirty-six', 37: 'thirty-seven', 38: 'thirty-eight', 39: 'thirty-nine', 40: 'forty',  
    41: 'forty-one', 42: 'forty-two', 43: 'forty-three', 44: 'forty-four', 45: 'forty-five',  
    46: 'forty-six', 47: 'forty-seven', 48: 'forty-eight', 49: 'forty-nine', 50: 'fifty',  
    51: 'fifty-one', 52: 'fifty-two', 53: 'fifty-three', 54: 'fifty-four', 55: 'fifty-five',  
    56: 'fifty-six', 57: 'fifty-seven', 58: 'fifty-eight', 59: 'fifty-nine', 60: 'sixty',  
    61: 'sixty-one', 62: 'sixty-two', 63: 'sixty-three', 64: 'sixty-four', 65: 'sixty-five',  
    66: 'sixty-six', 67: 'sixty-seven', 68: 'sixty-eight', 69: 'sixty-nine', 70: 'seventy',  
    71: 'seventy-one', 72: 'seventy-two', 73: 'seventy-three', 74: 'seventy-four', 75: 'seventy-five',  
    76: 'seventy-six', 77: 'seventy-seven', 78: 'seventy-eight', 79: 'seventy-nine', 80: 'eighty',  
    81: 'eighty-one', 82: 'eighty-two', 83: 'eighty-three', 84: 'eighty-four', 85: 'eighty-five',  
    86: 'eighty-six', 87: 'eighty-seven', 88: 'eighty-eight', 89: 'eighty-nine', 90: 'ninety',  
    91: 'ninety-one', 92: 'ninety-two', 93: 'ninety-three', 94: 'ninety-four', 95: 'ninety-five',  
    96: 'ninety-six', 97: 'ninety-seven', 98: 'ninety-eight', 99: 'ninety-nine', 100: 'one-hundred',
    101: 'one-hundred-one', 102: 'one-hundred-two', 103: 'one-hundred-three', 104: 'one-hundred-four', 105: 'one-hundred-five',
    106: 'one-hundred-six', 107: 'one-hundred-seven', 108: 'one-hundred-eight', 109: 'one-hundred-nine', 110: 'one-hundred-ten',
    111: 'one-hundred-eleven', 112: 'one-hundred-twelve', 113: 'one-hundred-thirteen', 114: 'one-hundred-fourteen', 115: 'one-hundred-fifteen',
    116: 'one-hundred-sixteen', 117: 'one-hundred-seventeen', 118: 'one-hundred-eighteen', 119: 'one-hundred-nineteen', 120: 'one-hundred-twenty',
    121: 'one-hundred-twenty-one', 122: 'one-hundred-twenty-two', 123: 'one-hundred-twenty-three', 124: 'one-hundred-twenty-four', 125: 'one-hundred-twenty-five',
    126: 'one-hundred-twenty-six', 127: 'one-hundred-twenty-seven', 128: 'one-hundred-twenty-eight', 129: 'one-hundred-twenty-nine', 130: 'one-hundred-thirty',
    131: 'one-hundred-thirty-one', 132: 'one-hundred-thirty-two', 133: 'one-hundred-thirty-three', 134: 'one-hundred-thirty-four', 135: 'one-hundred-thirty-five',
    136: 'one-hundred-thirty-six', 137: 'one-hundred-thirty-seven', 138: 'one-hundred-thirty-eight', 139: 'one-hundred-thirty-nine', 140: 'one-hundred-forty',
    141: 'one-hundred-forty-one', 142: 'one-hundred-forty-two', 143: 'one-hundred-forty-three', 144: 'one-hundred-forty-four', 145: 'one-hundred-forty-five',
    146: 'one-hundred-forty-six', 147: 'one-hundred-forty-seven', 148: 'one-hundred-forty-eight', 149: 'one-hundred-forty-nine', 150: 'one-hundred-fifty',
    151: 'one-hundred-fifty-one', 152: 'one-hundred-fifty-two', 153: 'one-hundred-fifty-three', 154: 'one-hundred-fifty-four', 155: 'one-hundred-fifty-five',
    156: 'one-hundred-fifty-six', 157: 'one-hundred-fifty-seven', 158: 'one-hundred-fifty-eight', 159: 'one-hundred-fifty-nine', 160: 'one-hundred-sixty',
    161: 'one-hundred-sixty-one', 162: 'one-hundred-sixty-two', 163: 'one-hundred-sixty-three', 164: 'one-hundred-sixty-four', 165: 'one-hundred-sixty-five',
    166: 'one-hundred-sixty-six', 167: 'one-hundred-sixty-seven', 168: 'one-hundred-sixty-eight', 169: 'one-hundred-sixty-nine', 170: 'one-hundred-seventy',
    171: 'one-hundred-seventy-one', 172: 'one-hundred-seventy-two', 173: 'one-hundred-seventy-three', 174: 'one-hundred-seventy-four', 175: 'one-hundred-seventy-five',
    176: 'one-hundred-seventy-six', 177: 'one-hundred-seventy-seven', 178: 'one-hundred-seventy-eight', 179: 'one-hundred-seventy-nine', 180: 'one-hundred-eighty',
    181: 'one-hundred-eighty-one', 182: 'one-hundred-eighty-two', 183: 'one-hundred-eighty-three', 184: 'one-hundred-eighty-four', 185: 'one-hundred-eighty-five',
    186: 'one-hundred-eighty-six', 187: 'one-hundred-eighty-seven', 188: 'one-hundred-eighty-eight', 189: 'one-hundred-eighty-nine', 190: 'one-hundred-ninety',
    191: 'one-hundred-ninety-one', 192: 'one-hundred-ninety-two', 193: 'one-hundred-ninety-three', 194: 'one-hundred-ninety-four', 195: 'one-hundred-ninety-five',
    196: 'one-hundred-ninety-six', 197: 'one-hundred-ninety-seven', 198: 'one-hundred-ninety-eight', 199: 'one-hundred-ninety-nine', 200: 'two-hundred'
}

MIN_SIZE = 5
MIRIDIH_TEMPLATE_TYPE={
    "instagram": [
        "instagram_feed",
        "instagram_story",
        "instagram_default"
    ],
    "card": [
        "business_card_hor",
        "business_card_ver",
        "card_news",
        "placard_square",
        "placard_hor",
        "placard_square_120_120_for_size",
        "placard_ver",
        "placard_poster_ver",
        "placard_poster_hor",
        "placard_square_120_120",
        "placard_hor_500_90_for_size",
        "post_card_152_102",
        "post_card_102_152",
        "post_card_90_90",
        "post_card_fold_ver",
        "post_card_fold_hor",
        "business_card_1color_ver",
        "business_card_1color_hor",
        "post_card_95_210",
        "photocard",
        "post_card_fold_hor_wide",
        "post_card_fold_ver_wide",
        "usb_card_hor",
        "usb_card_ver",
        "idcard_90_58",
        "idcard_58_90"
    ],
    "youtube": [
        "youtube_thumb",
        "youtube_cover"
    ],
    "brochure": [
        "brochure_catalogue_hor",
        "brochure_catalogue_ver",
        "brochure_catalogue_1to1"
    ],
    "other": [
        "led_panel_hor",
        "led_panel_ver",
        "corner_bag",
        "plastic_bag",
        "ticket_hor",
        "ticket_perforation_hor",
        "ticket_ver",
        "ticket_perforation_ver",
        "l_holder",
        "cocktail_napkin_square",
        "shoulder_sash",
        "groobee_big",
        "presentation",
        "detail_page",
        "pop_print_127_182",
        "pop_print_156_156",
        "pop_print_156_106",
        "hand_fan",
        "social_media_square",
        "general_document",
        "ticket_perforation_hor_M",
        "pop_print_182_257",
        "mobile_sub_img2",
        "mobile_sub_img",
        "acrylic_keyring_circle",
        "acrylic_keyring_square",
        "acrylic_keyring_heart",
        "acrylic_keyring_rectangle",
        "corrugated_box",
        "infographic",
        "apparel_1to1",
        "coaster_circle",
        "coaster_square",
        "mouse_pad_hor",
        "mouse_pad_ver",
        "masking_tape",
        "RCS_brand",
        "masking_tape_10to1",
        "masking_tape_8to1",
        "masking_tape_6to1",
        "masking_tape_4to1",
        "phone_case_hard",
        "phone_case_clear",
        "cocktail_napkin_rhombus",
        "clipen",
        "wetwipes_promotion",
        "clipboard_medium",
        "clipboard_small",
        "pop_print_297_420",
        "pop_print_customize",
        "pop_print_picket_circle",
        "pop_print_420_297",
        "pop_print_frame",
        "pop_print_picket_square",
        "hand_cream",
        "phone_loop_strap",
        "basicpen_9_1",
        "apparel_1color_1to1",
        "apparel_simple_1color_1to1",
        "toothbrush_sterilizer_ver",
        "toothbrush_sterilizer_hor",
        "toothbrush_sterilizer_hor_1color",
        "toothbrush_sterilizer_ver_1color",
        "picnic_mat_pouch",
        "picnic_mat",
        "picnic_mat_folding",
        "plastic_bag_mican",
        "balloon",
        "towel_dtg",
        "stamp_circle",
        "stamp_square",
        "stamp_hor",
        "window_decoration_lettering_ver",
        "window_decoration_lettering_hor",
        "opp_tape_10to1",
        "opp_tape_2to1",
        "drip_bag",
        "drip_bag_1color",
        "reusable_bottle_100_90",
        "reusable_bottle_100_130"
    ],
    "poster": [
        "poster_hor_print",
        "poster_ver_print",
        "web_post_ver_poster",
        "web_post_hor_poster",
        "poster_a3_hor_420_297_for_size",
        "clear_poster_ver",
        "clear_poster_hor",
        "stand_poster_leg_square",
        "stand_poster_leg_hor",
        "stand_poster_leg_ver"
    ],
    "banner": [
        "banner",
        "fancy_banner_circle",
        "fancy_banner_square",
        "fancy_banner_rect",
        "fancy_banner_oval",
        "placard_banner_ver",
        "banner_ver_60_180",
        "web_banner_hor2",
        "web_banner_hor",
        "samsunglife_banner",
        "web_banner_hor3",
        "wind_banner_s_small",
        "wind_banner_f_large",
        "wind_banner_f_medium",
        "wind_banner_f_small",
        "wind_banner_s_medium",
        "wind_banner_s_large"
    ],
    "flyer": [
        "web_flyer_double",
        "flyer_magnet_circle",
        "flyer_magnet_rounded_rect",
        "flyer_magnetic_paper",
        "web_flyer_single",
        "flyer_magnetic_single",
        "flyer_a4_ver_210_297_for_size_2",
        "flyer_ver_21_29.7",
        "flyer_posts",
        "flyer_doorknob"
    ],
    "logo": [
        "logo",
        "apparel_logo_ver",
        "apparel_logo_hor",
        "apparel_hat_logo",
        "vest_logo_ver",
        "vest_logo_hor",
        "apparel_logo_simple_1color_ver",
        "apparel_logo_simple_1color_hor",
        "promotional_item_logo_3_1",
        "promotional_item_logo_1_2",
        "promotional_item_logo_1_1",
        "promotional_item_logo_1_3",
        "umbrella_logo_3_1",
        "promotional_item_logo_2_1"
    ],
    "envelope": [
        "envelope_small",
        "envelope_big",
        "envelope_jacket",
        "envelope_ticket"
    ],
    "leaflet": [
        "leaflet_2_stages",
        "leaflet_3_stages",
        "leaflet_3_accordionfold",
        "leaflet_4_stages",
        "leaflet_4_gatefold",
        "leaflet_4_accordionfold"
    ],
    "hang_tag": [
        "hang_tag_1_2",
        "hang_tag_1_3",
        "hang_tag_square",
        "hang_tag_circle",
        "hang_tag_1color_1_2",
        "hang_tag_1color_1_3",
        "hang_tag_1color_square"
    ],
    "magnet": [
        "magnet_opener_pet",
        "car_magnet_standard",
        "car_magnet_wide",
        "car_magnet_vertical",
        "car_magnet_square"
    ],
    "sign": [
        "stand_sign_big",
        "stand_sign_normal",
        "stand_sign_small",
        "sign_board_circle",
        "sign_board_hexagon",
        "sign_board_square",
        "sign_board_rect_ver",
        "sign_board_rect_hor",
        "stainless_steel_stand_sign",
        "air_inflatable_sign_small",
        "air_inflatable_sign_wide",
        "air_inflatable_sign_large",
        "air_inflatable_sign_medium"
    ],
    "paper": [
        "table_paper",
        "wallpaper_pc",
        "wallpaper_mobile",
        "shopping_bag_paper_small",
        "shopping_bag_paper_medium",
        "shopping_bag_paper_large",
        "ricecake_memo_paper_square",
        "ricecake_memo_paper_hor_rect",
        "ricecake_memo_paper_ver_rect",
        "shopping_bag_paper_gift_small",
        "shopping_bag_paper_gift_large",
        "paper_cup_16",
        "paper_cup_10",
        "paper_cup_13",
        "paper_cup_6_5",
        "paper_container_850",
        "paper_container_350",
        "paper_container_1000",
        "paper_container_1200",
        "acrylic_keyring_papercup",
        "double_wall_paper_cup_10",
        "double_wall_paper_cup_12",
        "double_wall_paper_cup_13",
        "double_wall_paper_cup_16",
        "paper_sun_cap",
        "shopping_bag_paper_kraft_large",
        "shopping_bag_paper_kraft_medium",
        "shopping_bag_paper_kraft_small",
        "format_paper_ver_1_2",
        "format_paper_ver_a4",
        "format_paper_hor_a4",
        "format_paper_hor_2_1"
    ],
    "book": [
        "coloring_book",
        "book_cover",
        "facebook_default",
        "book_ver",
        "facebook_video_horizontal",
        "facebook_video_square",
        "facebook_video_story"
    ],
    "sticker": [
        "sticker_circle",
        "sticker_rect",
        "sticker_vertical_rect",
        "sticker_box_packing",
        "sticker_gift_wrapping_ver",
        "sticker_container_packing_big",
        "sticker_gift_wrapping",
        "sticker_container_packing_big_ver",
        "sticker_container_packing_normal",
        "sticker_container_packing_normal_ver",
        "sticker_square",
        "sticker_a4_normal",
        "sticker_a4_clear",
        "sticker_tattoo",
        "interior_sticker_hor",
        "interior_sticker_ver",
        "interior_sticker_1row",
        "sticker_single_square",
        "sticker_single_hor",
        "sticker_single_ver",
        "print_sticker_square",
        "print_sticker_ver",
        "print_sticker_ver_wide",
        "information_sticker_hor_wide",
        "information_sticker_ver_wide",
        "information_sticker_hor",
        "information_sticker_ver",
        "information_sticker_1to1"
    ],
    "post": [
        "web_post_square",
        "web_post_ver",
        "instagram_video_post"
    ],
    "cup": [
        "cup_holder_10_13",
        "cup_holder_12_16",
        "solid_cup_holder_10_13",
        "pet_ice_cup",
        "solid_cup_holder_12_16",
        "acrylic_keyring_icecup",
        "mug_cup_11",
        "cup_carrier_kraft",
        "pet_ice_cup_hor",
        "pet_ice_cup_ver",
        "mug_cup_1to1_1color",
        "mug_cup_1to1",
        "mug_cup_hor",
        "mug_cup_hor_1color"
    ],
    "mug": [
        "acrylic_keyring_mug"
    ],
    "tshirts": [
        "tshirts_half_hor",
        "tshirts_half_ver",
        "tshirts_half_1row",
        "tshirts_half_dtg"
    ],
    "stand": [
        "stand_nameplate",
        "stand_freeshape_ver",
        "stand_freeshape_1to1",
        "stand_freeshape_multi",
        "stand_pouch",
        "stand_pouch_1color"
    ],
    "vest": [
        "vest_1row",
        "vest_ver",
        "vest_hor"
    ]
}
