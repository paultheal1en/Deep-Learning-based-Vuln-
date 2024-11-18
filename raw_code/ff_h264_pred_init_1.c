av_cold void ff_h264_pred_init(H264PredContext *h, int codec_id,

                               const int bit_depth,

                               const int chroma_format_idc)

{

#undef FUNC

#undef FUNCC

#define FUNC(a, depth) a ## _ ## depth

#define FUNCC(a, depth) a ## _ ## depth ## _c

#define FUNCD(a) a ## _c



#define H264_PRED(depth) \

    if(codec_id != AV_CODEC_ID_RV40){\

        if(codec_id == AV_CODEC_ID_VP8) {\

            h->pred4x4[VERT_PRED       ]= FUNCD(pred4x4_vertical_vp8);\

            h->pred4x4[HOR_PRED        ]= FUNCD(pred4x4_horizontal_vp8);\

        } else {\

            h->pred4x4[VERT_PRED       ]= FUNCC(pred4x4_vertical          , depth);\

            h->pred4x4[HOR_PRED        ]= FUNCC(pred4x4_horizontal        , depth);\

        }\

        h->pred4x4[DC_PRED             ]= FUNCC(pred4x4_dc                , depth);\

        if(codec_id == AV_CODEC_ID_SVQ3)\

            h->pred4x4[DIAG_DOWN_LEFT_PRED ]= FUNCD(pred4x4_down_left_svq3);\

        else\

            h->pred4x4[DIAG_DOWN_LEFT_PRED ]= FUNCC(pred4x4_down_left     , depth);\

        h->pred4x4[DIAG_DOWN_RIGHT_PRED]= FUNCC(pred4x4_down_right        , depth);\

        h->pred4x4[VERT_RIGHT_PRED     ]= FUNCC(pred4x4_vertical_right    , depth);\

        h->pred4x4[HOR_DOWN_PRED       ]= FUNCC(pred4x4_horizontal_down   , depth);\

        if (codec_id == AV_CODEC_ID_VP8) {\

            h->pred4x4[VERT_LEFT_PRED  ]= FUNCD(pred4x4_vertical_left_vp8);\

        } else\

            h->pred4x4[VERT_LEFT_PRED  ]= FUNCC(pred4x4_vertical_left     , depth);\

        h->pred4x4[HOR_UP_PRED         ]= FUNCC(pred4x4_horizontal_up     , depth);\

        if(codec_id != AV_CODEC_ID_VP8) {\

            h->pred4x4[LEFT_DC_PRED    ]= FUNCC(pred4x4_left_dc           , depth);\

            h->pred4x4[TOP_DC_PRED     ]= FUNCC(pred4x4_top_dc            , depth);\

            h->pred4x4[DC_128_PRED     ]= FUNCC(pred4x4_128_dc            , depth);\

        } else {\

            h->pred4x4[TM_VP8_PRED     ]= FUNCD(pred4x4_tm_vp8);\

            h->pred4x4[DC_127_PRED     ]= FUNCC(pred4x4_127_dc            , depth);\

            h->pred4x4[DC_129_PRED     ]= FUNCC(pred4x4_129_dc            , depth);\

            h->pred4x4[VERT_VP8_PRED   ]= FUNCC(pred4x4_vertical          , depth);\

            h->pred4x4[HOR_VP8_PRED    ]= FUNCC(pred4x4_horizontal        , depth);\

        }\

    }else{\

        h->pred4x4[VERT_PRED           ]= FUNCC(pred4x4_vertical          , depth);\

        h->pred4x4[HOR_PRED            ]= FUNCC(pred4x4_horizontal        , depth);\

        h->pred4x4[DC_PRED             ]= FUNCC(pred4x4_dc                , depth);\

        h->pred4x4[DIAG_DOWN_LEFT_PRED ]= FUNCD(pred4x4_down_left_rv40);\

        h->pred4x4[DIAG_DOWN_RIGHT_PRED]= FUNCC(pred4x4_down_right        , depth);\

        h->pred4x4[VERT_RIGHT_PRED     ]= FUNCC(pred4x4_vertical_right    , depth);\

        h->pred4x4[HOR_DOWN_PRED       ]= FUNCC(pred4x4_horizontal_down   , depth);\

        h->pred4x4[VERT_LEFT_PRED      ]= FUNCD(pred4x4_vertical_left_rv40);\

        h->pred4x4[HOR_UP_PRED         ]= FUNCD(pred4x4_horizontal_up_rv40);\

        h->pred4x4[LEFT_DC_PRED        ]= FUNCC(pred4x4_left_dc           , depth);\

        h->pred4x4[TOP_DC_PRED         ]= FUNCC(pred4x4_top_dc            , depth);\

        h->pred4x4[DC_128_PRED         ]= FUNCC(pred4x4_128_dc            , depth);\

        h->pred4x4[DIAG_DOWN_LEFT_PRED_RV40_NODOWN]= FUNCD(pred4x4_down_left_rv40_nodown);\

        h->pred4x4[HOR_UP_PRED_RV40_NODOWN]= FUNCD(pred4x4_horizontal_up_rv40_nodown);\

        h->pred4x4[VERT_LEFT_PRED_RV40_NODOWN]= FUNCD(pred4x4_vertical_left_rv40_nodown);\

    }\

\

    h->pred8x8l[VERT_PRED           ]= FUNCC(pred8x8l_vertical            , depth);\

    h->pred8x8l[HOR_PRED            ]= FUNCC(pred8x8l_horizontal          , depth);\

    h->pred8x8l[DC_PRED             ]= FUNCC(pred8x8l_dc                  , depth);\

    h->pred8x8l[DIAG_DOWN_LEFT_PRED ]= FUNCC(pred8x8l_down_left           , depth);\

    h->pred8x8l[DIAG_DOWN_RIGHT_PRED]= FUNCC(pred8x8l_down_right          , depth);\

    h->pred8x8l[VERT_RIGHT_PRED     ]= FUNCC(pred8x8l_vertical_right      , depth);\

    h->pred8x8l[HOR_DOWN_PRED       ]= FUNCC(pred8x8l_horizontal_down     , depth);\

    h->pred8x8l[VERT_LEFT_PRED      ]= FUNCC(pred8x8l_vertical_left       , depth);\

    h->pred8x8l[HOR_UP_PRED         ]= FUNCC(pred8x8l_horizontal_up       , depth);\

    h->pred8x8l[LEFT_DC_PRED        ]= FUNCC(pred8x8l_left_dc             , depth);\

    h->pred8x8l[TOP_DC_PRED         ]= FUNCC(pred8x8l_top_dc              , depth);\

    h->pred8x8l[DC_128_PRED         ]= FUNCC(pred8x8l_128_dc              , depth);\

\

    if (chroma_format_idc <= 1) {\

        h->pred8x8[VERT_PRED8x8   ]= FUNCC(pred8x8_vertical               , depth);\

        h->pred8x8[HOR_PRED8x8    ]= FUNCC(pred8x8_horizontal             , depth);\

    } else {\

        h->pred8x8[VERT_PRED8x8   ]= FUNCC(pred8x16_vertical              , depth);\

        h->pred8x8[HOR_PRED8x8    ]= FUNCC(pred8x16_horizontal            , depth);\

    }\

    if (codec_id != AV_CODEC_ID_VP8) {\

        if (chroma_format_idc <= 1) {\

            h->pred8x8[PLANE_PRED8x8]= FUNCC(pred8x8_plane                , depth);\

        } else {\

            h->pred8x8[PLANE_PRED8x8]= FUNCC(pred8x16_plane               , depth);\

        }\

    } else\

        h->pred8x8[PLANE_PRED8x8]= FUNCD(pred8x8_tm_vp8);\

    if(codec_id != AV_CODEC_ID_RV40 && codec_id != AV_CODEC_ID_VP8){\

        if (chroma_format_idc <= 1) {\

            h->pred8x8[DC_PRED8x8     ]= FUNCC(pred8x8_dc                     , depth);\

            h->pred8x8[LEFT_DC_PRED8x8]= FUNCC(pred8x8_left_dc                , depth);\

            h->pred8x8[TOP_DC_PRED8x8 ]= FUNCC(pred8x8_top_dc                 , depth);\

            h->pred8x8[ALZHEIMER_DC_L0T_PRED8x8 ]= FUNC(pred8x8_mad_cow_dc_l0t, depth);\

            h->pred8x8[ALZHEIMER_DC_0LT_PRED8x8 ]= FUNC(pred8x8_mad_cow_dc_0lt, depth);\

            h->pred8x8[ALZHEIMER_DC_L00_PRED8x8 ]= FUNC(pred8x8_mad_cow_dc_l00, depth);\

            h->pred8x8[ALZHEIMER_DC_0L0_PRED8x8 ]= FUNC(pred8x8_mad_cow_dc_0l0, depth);\

        } else {\

            h->pred8x8[DC_PRED8x8     ]= FUNCC(pred8x16_dc                    , depth);\

            h->pred8x8[LEFT_DC_PRED8x8]= FUNCC(pred8x16_left_dc               , depth);\

            h->pred8x8[TOP_DC_PRED8x8 ]= FUNCC(pred8x16_top_dc                , depth);\

            h->pred8x8[ALZHEIMER_DC_L0T_PRED8x8 ]= FUNC(pred8x16_mad_cow_dc_l0t, depth);\

            h->pred8x8[ALZHEIMER_DC_0LT_PRED8x8 ]= FUNC(pred8x16_mad_cow_dc_0lt, depth);\

            h->pred8x8[ALZHEIMER_DC_L00_PRED8x8 ]= FUNC(pred8x16_mad_cow_dc_l00, depth);\

            h->pred8x8[ALZHEIMER_DC_0L0_PRED8x8 ]= FUNC(pred8x16_mad_cow_dc_0l0, depth);\

        }\

    }else{\

        h->pred8x8[DC_PRED8x8     ]= FUNCD(pred8x8_dc_rv40);\

        h->pred8x8[LEFT_DC_PRED8x8]= FUNCD(pred8x8_left_dc_rv40);\

        h->pred8x8[TOP_DC_PRED8x8 ]= FUNCD(pred8x8_top_dc_rv40);\

        if (codec_id == AV_CODEC_ID_VP8) {\

            h->pred8x8[DC_127_PRED8x8]= FUNCC(pred8x8_127_dc              , depth);\

            h->pred8x8[DC_129_PRED8x8]= FUNCC(pred8x8_129_dc              , depth);\

        }\

    }\

    if (chroma_format_idc <= 1) {\

        h->pred8x8[DC_128_PRED8x8 ]= FUNCC(pred8x8_128_dc                 , depth);\

    } else {\

        h->pred8x8[DC_128_PRED8x8 ]= FUNCC(pred8x16_128_dc                , depth);\

    }\

\

    h->pred16x16[DC_PRED8x8     ]= FUNCC(pred16x16_dc                     , depth);\

    h->pred16x16[VERT_PRED8x8   ]= FUNCC(pred16x16_vertical               , depth);\

    h->pred16x16[HOR_PRED8x8    ]= FUNCC(pred16x16_horizontal             , depth);\

    switch(codec_id){\

    case AV_CODEC_ID_SVQ3:\

       h->pred16x16[PLANE_PRED8x8  ]= FUNCD(pred16x16_plane_svq3);\

       break;\

    case AV_CODEC_ID_RV40:\

       h->pred16x16[PLANE_PRED8x8  ]= FUNCD(pred16x16_plane_rv40);\

       break;\

    case AV_CODEC_ID_VP8:\

       h->pred16x16[PLANE_PRED8x8  ]= FUNCD(pred16x16_tm_vp8);\

       h->pred16x16[DC_127_PRED8x8]= FUNCC(pred16x16_127_dc               , depth);\

       h->pred16x16[DC_129_PRED8x8]= FUNCC(pred16x16_129_dc               , depth);\

       break;\

    default:\

       h->pred16x16[PLANE_PRED8x8  ]= FUNCC(pred16x16_plane               , depth);\

       break;\

    }\

    h->pred16x16[LEFT_DC_PRED8x8]= FUNCC(pred16x16_left_dc                , depth);\

    h->pred16x16[TOP_DC_PRED8x8 ]= FUNCC(pred16x16_top_dc                 , depth);\

    h->pred16x16[DC_128_PRED8x8 ]= FUNCC(pred16x16_128_dc                 , depth);\

\

    /* special lossless h/v prediction for h264 */ \

    h->pred4x4_add  [VERT_PRED   ]= FUNCC(pred4x4_vertical_add            , depth);\

    h->pred4x4_add  [ HOR_PRED   ]= FUNCC(pred4x4_horizontal_add          , depth);\

    h->pred8x8l_add [VERT_PRED   ]= FUNCC(pred8x8l_vertical_add           , depth);\

    h->pred8x8l_add [ HOR_PRED   ]= FUNCC(pred8x8l_horizontal_add         , depth);\

    if (chroma_format_idc <= 1) {\

    h->pred8x8_add  [VERT_PRED8x8]= FUNCC(pred8x8_vertical_add            , depth);\

    h->pred8x8_add  [ HOR_PRED8x8]= FUNCC(pred8x8_horizontal_add          , depth);\

    } else {\

        h->pred8x8_add  [VERT_PRED8x8]= FUNCC(pred8x16_vertical_add            , depth);\

        h->pred8x8_add  [ HOR_PRED8x8]= FUNCC(pred8x16_horizontal_add          , depth);\

    }\

    h->pred16x16_add[VERT_PRED8x8]= FUNCC(pred16x16_vertical_add          , depth);\

    h->pred16x16_add[ HOR_PRED8x8]= FUNCC(pred16x16_horizontal_add        , depth);\



    switch (bit_depth) {

        case 9:

            H264_PRED(9)

            break;

        case 10:

            H264_PRED(10)

            break;

        default:

            H264_PRED(8)

            break;

    }



    if (ARCH_ARM) ff_h264_pred_init_arm(h, codec_id, bit_depth, chroma_format_idc);

    if (ARCH_X86) ff_h264_pred_init_x86(h, codec_id, bit_depth, chroma_format_idc);

}
