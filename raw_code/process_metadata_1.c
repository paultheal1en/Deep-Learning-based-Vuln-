static int process_metadata(AVFormatContext *s, uint8_t *name, uint16_t name_len,

                            uint16_t val_len, uint16_t type, AVDictionary **met)

{

    int ret;

    ff_asf_guid guid;



    if (val_len) {

        switch (type) {

        case ASF_UNICODE:

            asf_read_value(s, name, name_len, val_len, type, met);

            break;

        case ASF_BYTE_ARRAY:

            if (!strcmp(name, "WM/Picture")) // handle cover art

                asf_read_picture(s, val_len);

            else if (!strcmp(name, "ID3")) // handle ID3 tag

                get_id3_tag(s, val_len);

            else

                asf_read_value(s, name, name_len, val_len, type, met);

            break;

        case ASF_GUID:

            ff_get_guid(s->pb, &guid);

            break;

        default:

            if ((ret = asf_read_generic_value(s, name, name_len, type, met)) < 0)

                return ret;

            break;

        }

    }

    av_freep(&name);



    return 0;

}
