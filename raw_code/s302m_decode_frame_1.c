static int s302m_decode_frame(AVCodecContext *avctx, void *data,

                              int *got_frame_ptr, AVPacket *avpkt)

{

    S302Context *s = avctx->priv_data;

    AVFrame *frame     = data;

    const uint8_t *buf = avpkt->data;

    int buf_size       = avpkt->size;

    int block_size, ret;

    int i;

    int non_pcm_data_type = -1;



    int frame_size = s302m_parse_frame_header(avctx, buf, buf_size);

    if (frame_size < 0)

        return frame_size;



    buf_size -= AES3_HEADER_LEN;

    buf      += AES3_HEADER_LEN;



    /* get output buffer */

    block_size = (avctx->bits_per_raw_sample + 4) / 4;

    frame->nb_samples = 2 * (buf_size / block_size) / avctx->channels;

    if ((ret = ff_get_buffer(avctx, frame, 0)) < 0)

        return ret;



    avctx->bit_rate = 48000 * avctx->channels * (avctx->bits_per_raw_sample + 4) +

                      32 * 48000 / frame->nb_samples;

    buf_size = (frame->nb_samples * avctx->channels / 2) * block_size;



    if (avctx->bits_per_raw_sample == 24) {

        uint32_t *o = (uint32_t *)frame->data[0];

        for (; buf_size > 6; buf_size -= 7) {

            *o++ = (ff_reverse[buf[2]]        << 24) |

                   (ff_reverse[buf[1]]        << 16) |

                   (ff_reverse[buf[0]]        <<  8);

            *o++ = (ff_reverse[buf[6] & 0xf0] << 28) |

                   (ff_reverse[buf[5]]        << 20) |

                   (ff_reverse[buf[4]]        << 12) |

                   (ff_reverse[buf[3] & 0x0f] <<  4);

            buf += 7;

        }

        o = (uint32_t *)frame->data[0];

        if (avctx->channels == 2)

            for (i=0; i<frame->nb_samples * 2 - 6; i+=2) {

                if (o[i] || o[i+1] || o[i+2] || o[i+3])

                    break;

                if (o[i+4] == 0x96F87200U && o[i+5] == 0xA54E1F00) {

                    non_pcm_data_type = (o[i+6] >> 16) & 0x1F;

                    break;

                }

            }

    } else if (avctx->bits_per_raw_sample == 20) {

        uint32_t *o = (uint32_t *)frame->data[0];

        for (; buf_size > 5; buf_size -= 6) {

            *o++ = (ff_reverse[buf[2] & 0xf0] << 28) |

                   (ff_reverse[buf[1]]        << 20) |

                   (ff_reverse[buf[0]]        << 12);

            *o++ = (ff_reverse[buf[5] & 0xf0] << 28) |

                   (ff_reverse[buf[4]]        << 20) |

                   (ff_reverse[buf[3]]        << 12);

            buf += 6;

        }

        o = (uint32_t *)frame->data[0];

        if (avctx->channels == 2)

            for (i=0; i<frame->nb_samples * 2 - 6; i+=2) {

                if (o[i] || o[i+1] || o[i+2] || o[i+3])

                    break;

                if (o[i+4] == 0x6F872000U && o[i+5] == 0x54E1F000) {

                    non_pcm_data_type = (o[i+6] >> 16) & 0x1F;

                    break;

                }

            }

    } else {

        uint16_t *o = (uint16_t *)frame->data[0];

        for (; buf_size > 4; buf_size -= 5) {

            *o++ = (ff_reverse[buf[1]]        <<  8) |

                    ff_reverse[buf[0]];

            *o++ = (ff_reverse[buf[4] & 0xf0] << 12) |

                   (ff_reverse[buf[3]]        <<  4) |

                   (ff_reverse[buf[2]]        >>  4);

            buf += 5;

        }

        o = (uint16_t *)frame->data[0];

        if (avctx->channels == 2)

            for (i=0; i<frame->nb_samples * 2 - 6; i+=2) {

                if (o[i] || o[i+1] || o[i+2] || o[i+3])

                    break;

                if (o[i+4] == 0xF872U && o[i+5] == 0x4E1F) {

                    non_pcm_data_type = (o[i+6] & 0x1F);

                    break;

                }

            }

    }



    if (non_pcm_data_type != -1) {

        if (s->non_pcm_mode == 3) {

            av_log(avctx, AV_LOG_ERROR,

                   "S302 non PCM mode with data type %d not supported\n",

                   non_pcm_data_type);

            return AVERROR_PATCHWELCOME;

        }

        if (s->non_pcm_mode & 1) {

            return avpkt->size;

        }

    }



    avctx->sample_rate = 48000;



    *got_frame_ptr = 1;



    return avpkt->size;

}
