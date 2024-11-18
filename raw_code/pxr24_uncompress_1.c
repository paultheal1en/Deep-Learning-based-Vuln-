static int pxr24_uncompress(EXRContext *s, const uint8_t *src,

                            int compressed_size, int uncompressed_size,

                            EXRThreadData *td)

{

    unsigned long dest_len, expected_len = 0;

    const uint8_t *in = td->tmp;

    uint8_t *out;

    int c, i, j;



    for (i = 0; i < s->nb_channels; i++) {

        if (s->channels[i].pixel_type == EXR_FLOAT) {

            expected_len += (td->xsize * td->ysize * 3);/* PRX 24 store float in 24 bit instead of 32 */

        } else if (s->channels[i].pixel_type == EXR_HALF) {

            expected_len += (td->xsize * td->ysize * 2);

        } else {//UINT 32

            expected_len += (td->xsize * td->ysize * 4);

        }

    }



    dest_len = expected_len;



    if (uncompress(td->tmp, &dest_len, src, compressed_size) != Z_OK) {

        return AVERROR_INVALIDDATA;

    } else if (dest_len != expected_len) {

        return AVERROR_INVALIDDATA;

    }



    out = td->uncompressed_data;

    for (i = 0; i < td->ysize; i++)

        for (c = 0; c < s->nb_channels; c++) {

            EXRChannel *channel = &s->channels[c];

            const uint8_t *ptr[4];

            uint32_t pixel = 0;



            switch (channel->pixel_type) {

            case EXR_FLOAT:

                ptr[0] = in;

                ptr[1] = ptr[0] + td->xsize;

                ptr[2] = ptr[1] + td->xsize;

                in     = ptr[2] + td->xsize;



                for (j = 0; j < td->xsize; ++j) {

                    uint32_t diff = (*(ptr[0]++) << 24) |

                                    (*(ptr[1]++) << 16) |

                                    (*(ptr[2]++) << 8);

                    pixel += diff;

                    bytestream_put_le32(&out, pixel);

                }

                break;

            case EXR_HALF:

                ptr[0] = in;

                ptr[1] = ptr[0] + td->xsize;

                in     = ptr[1] + td->xsize;

                for (j = 0; j < td->xsize; j++) {

                    uint32_t diff = (*(ptr[0]++) << 8) | *(ptr[1]++);



                    pixel += diff;

                    bytestream_put_le16(&out, pixel);

                }

                break;

            case EXR_UINT:

                ptr[0] = in;

                ptr[1] = ptr[0] + s->xdelta;

                ptr[2] = ptr[1] + s->xdelta;

                ptr[3] = ptr[2] + s->xdelta;

                in     = ptr[3] + s->xdelta;



                for (j = 0; j < s->xdelta; ++j) {

                    uint32_t diff = (*(ptr[0]++) << 24) |

                    (*(ptr[1]++) << 16) |

                    (*(ptr[2]++) << 8 ) |

                    (*(ptr[3]++));

                    pixel += diff;

                    bytestream_put_le32(&out, pixel);

                }

                break;

            default:

                return AVERROR_INVALIDDATA;

            }

        }



    return 0;

}
