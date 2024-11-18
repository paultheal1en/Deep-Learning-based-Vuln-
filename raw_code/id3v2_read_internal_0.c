static void id3v2_read_internal(AVIOContext *pb, AVDictionary **metadata,

                                AVFormatContext *s, const char *magic,

                                ID3v2ExtraMeta **extra_meta)

{

    int len, ret;

    uint8_t buf[ID3v2_HEADER_SIZE];

    int found_header;

    int64_t off;



    do {

        /* save the current offset in case there's nothing to read/skip */

        off = avio_tell(pb);

        ret = avio_read(pb, buf, ID3v2_HEADER_SIZE);

        if (ret != ID3v2_HEADER_SIZE) {

            avio_seek(pb, off, SEEK_SET);

            break;

        }

        found_header = ff_id3v2_match(buf, magic);

        if (found_header) {

            /* parse ID3v2 header */

            len = ((buf[6] & 0x7f) << 21) |

                  ((buf[7] & 0x7f) << 14) |

                  ((buf[8] & 0x7f) << 7) |

                   (buf[9] & 0x7f);

            id3v2_parse(pb, metadata, s, len, buf[3], buf[5], extra_meta);

        } else {

            avio_seek(pb, off, SEEK_SET);

        }

    } while (found_header);

    ff_metadata_conv(metadata, NULL, ff_id3v2_34_metadata_conv);

    ff_metadata_conv(metadata, NULL, id3v2_2_metadata_conv);

    ff_metadata_conv(metadata, NULL, ff_id3v2_4_metadata_conv);

    merge_date(metadata);

}
