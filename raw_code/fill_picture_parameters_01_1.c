static void fill_picture_parameters(AVCodecContext *avctx,

                                    struct dxva_context *ctx, const VC1Context *v,

                                    DXVA_PictureParameters *pp)

{

    const MpegEncContext *s = &v->s;

    const Picture *current_picture = s->current_picture_ptr;



    memset(pp, 0, sizeof(*pp));

    pp->wDecodedPictureIndex    =

    pp->wDeblockedPictureIndex  = ff_dxva2_get_surface_index(ctx, &current_picture->f);

    if (s->pict_type != AV_PICTURE_TYPE_I && !v->bi_type)

        pp->wForwardRefPictureIndex = ff_dxva2_get_surface_index(ctx, &s->last_picture.f);

    else

        pp->wForwardRefPictureIndex = 0xffff;

    if (s->pict_type == AV_PICTURE_TYPE_B && !v->bi_type)

        pp->wBackwardRefPictureIndex = ff_dxva2_get_surface_index(ctx, &s->next_picture.f);

    else

        pp->wBackwardRefPictureIndex = 0xffff;

    if (v->profile == PROFILE_ADVANCED) {

        /* It is the cropped width/height -1 of the frame */

        pp->wPicWidthInMBminus1 = avctx->width  - 1;

        pp->wPicHeightInMBminus1= avctx->height - 1;

    } else {

        /* It is the coded width/height in macroblock -1 of the frame */

        pp->wPicWidthInMBminus1 = s->mb_width  - 1;

        pp->wPicHeightInMBminus1= s->mb_height - 1;

    }

    pp->bMacroblockWidthMinus1  = 15;

    pp->bMacroblockHeightMinus1 = 15;

    pp->bBlockWidthMinus1       = 7;

    pp->bBlockHeightMinus1      = 7;

    pp->bBPPminus1              = 7;

    if (s->picture_structure & PICT_TOP_FIELD)

        pp->bPicStructure      |= 0x01;

    if (s->picture_structure & PICT_BOTTOM_FIELD)

        pp->bPicStructure      |= 0x02;

    pp->bSecondField            = v->interlace && v->fcm != ILACE_FIELD && !s->first_field;

    pp->bPicIntra               = s->pict_type == AV_PICTURE_TYPE_I || v->bi_type;

    pp->bPicBackwardPrediction  = s->pict_type == AV_PICTURE_TYPE_B && !v->bi_type;

    pp->bBidirectionalAveragingMode = (1                                           << 7) |

                                      ((ctx->cfg->ConfigIntraResidUnsigned != 0)   << 6) |

                                      ((ctx->cfg->ConfigResidDiffAccelerator != 0) << 5) |

                                      ((v->lumscale != 32 || v->lumshift != 0)     << 4) |

                                      ((v->profile == PROFILE_ADVANCED)            << 3);

    pp->bMVprecisionAndChromaRelation = ((v->mv_mode == MV_PMODE_1MV_HPEL_BILIN) << 3) |

                                        (1                                       << 2) |

                                        (0                                       << 1) |

                                        (!s->quarter_sample                          );

    pp->bChromaFormat           = v->chromaformat;

    ctx->report_id++;

    if (ctx->report_id >= (1 << 16))

        ctx->report_id = 1;

    pp->bPicScanFixed           = ctx->report_id >> 8;

    pp->bPicScanMethod          = ctx->report_id & 0xff;

    pp->bPicReadbackRequests    = 0;

    pp->bRcontrol               = v->rnd;

    pp->bPicSpatialResid8       = (v->panscanflag  << 7) |

                                  (v->refdist_flag << 6) |

                                  (s->loop_filter  << 5) |

                                  (v->fastuvmc     << 4) |

                                  (v->extended_mv  << 3) |

                                  (v->dquant       << 1) |

                                  (v->vstransform      );

    pp->bPicOverflowBlocks      = (v->quantizer_mode << 6) |

                                  (v->multires       << 5) |

                                  (v->resync_marker  << 4) |

                                  (v->rangered       << 3) |

                                  (s->max_b_frames       );

    pp->bPicExtrapolation       = (!v->interlace || v->fcm == PROGRESSIVE) ? 1 : 2;

    pp->bPicDeblocked           = ((!pp->bPicBackwardPrediction && v->overlap)        << 6) |

                                  ((v->profile != PROFILE_ADVANCED && v->rangeredfrm) << 5) |

                                  (s->loop_filter                                     << 1);

    pp->bPicDeblockConfined     = (v->postprocflag             << 7) |

                                  (v->broadcast                << 6) |

                                  (v->interlace                << 5) |

                                  (v->tfcntrflag               << 4) |

                                  (v->finterpflag              << 3) |

                                  ((s->pict_type != AV_PICTURE_TYPE_B) << 2) |

                                  (v->psf                      << 1) |

                                  (v->extended_dmv                 );

    if (s->pict_type != AV_PICTURE_TYPE_I)

        pp->bPic4MVallowed      = v->mv_mode == MV_PMODE_MIXED_MV ||

                                  (v->mv_mode == MV_PMODE_INTENSITY_COMP &&

                                   v->mv_mode2 == MV_PMODE_MIXED_MV);

    if (v->profile == PROFILE_ADVANCED)

        pp->bPicOBMC            = (v->range_mapy_flag  << 7) |

                                  (v->range_mapy       << 4) |

                                  (v->range_mapuv_flag << 3) |

                                  (v->range_mapuv          );

    pp->bPicBinPB               = 0;

    pp->bMV_RPS                 = 0;

    pp->bReservedBits           = 0;

    if (s->picture_structure == PICT_FRAME) {

        pp->wBitstreamFcodes        = v->lumscale;

        pp->wBitstreamPCEelements   = v->lumshift;

    } else {

        /* Syntax: (top_field_param << 8) | bottom_field_param */

        pp->wBitstreamFcodes        = (v->lumscale << 8) | v->lumscale;

        pp->wBitstreamPCEelements   = (v->lumshift << 8) | v->lumshift;

    }

    pp->bBitstreamConcealmentNeed   = 0;

    pp->bBitstreamConcealmentMethod = 0;

}
