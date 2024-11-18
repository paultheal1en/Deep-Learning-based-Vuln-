static int amf_parse_object(AVFormatContext *s, AVStream *astream, AVStream *vstream, const char *key, int64_t max_pos, int depth) {

    AVCodecContext *acodec, *vcodec;

    ByteIOContext *ioc;

    AMFDataType amf_type;

    char str_val[256];

    double num_val;



    num_val = 0;

    ioc = s->pb;



    amf_type = get_byte(ioc);



    switch(amf_type) {

        case AMF_DATA_TYPE_NUMBER:

            num_val = av_int2dbl(get_be64(ioc)); break;

        case AMF_DATA_TYPE_BOOL:

            num_val = get_byte(ioc); break;

        case AMF_DATA_TYPE_STRING:

            if(amf_get_string(ioc, str_val, sizeof(str_val)) < 0)

                return -1;

            break;

        case AMF_DATA_TYPE_OBJECT: {

            unsigned int keylen;



            while(url_ftell(ioc) < max_pos - 2 && (keylen = get_be16(ioc))) {

                url_fskip(ioc, keylen); //skip key string

                if(amf_parse_object(s, NULL, NULL, NULL, max_pos, depth + 1) < 0)

                    return -1; //if we couldn't skip, bomb out.

            }

            if(get_byte(ioc) != AMF_END_OF_OBJECT)

                return -1;

        }

            break;

        case AMF_DATA_TYPE_NULL:

        case AMF_DATA_TYPE_UNDEFINED:

        case AMF_DATA_TYPE_UNSUPPORTED:

            break; //these take up no additional space

        case AMF_DATA_TYPE_MIXEDARRAY:

            url_fskip(ioc, 4); //skip 32-bit max array index

            while(url_ftell(ioc) < max_pos - 2 && amf_get_string(ioc, str_val, sizeof(str_val)) > 0) {

                //this is the only case in which we would want a nested parse to not skip over the object

                if(amf_parse_object(s, astream, vstream, str_val, max_pos, depth + 1) < 0)

                    return -1;

            }

            if(get_byte(ioc) != AMF_END_OF_OBJECT)

                return -1;

            break;

        case AMF_DATA_TYPE_ARRAY: {

            unsigned int arraylen, i;



            arraylen = get_be32(ioc);

            for(i = 0; i < arraylen && url_ftell(ioc) < max_pos - 1; i++) {

                if(amf_parse_object(s, NULL, NULL, NULL, max_pos, depth + 1) < 0)

                    return -1; //if we couldn't skip, bomb out.

            }

        }

            break;

        case AMF_DATA_TYPE_DATE:

            url_fskip(ioc, 8 + 2); //timestamp (double) and UTC offset (int16)

            break;

        default: //unsupported type, we couldn't skip

            return -1;

    }



    if(depth == 1 && key) { //only look for metadata values when we are not nested and key != NULL

        acodec = astream ? astream->codec : NULL;

        vcodec = vstream ? vstream->codec : NULL;



        if(amf_type == AMF_DATA_TYPE_BOOL) {

        } else if(amf_type == AMF_DATA_TYPE_NUMBER) {

            if(!strcmp(key, "duration")) s->duration = num_val * AV_TIME_BASE;

            else if(!strcmp(key, "videodatarate") && vcodec && 0 <= (int)(num_val * 1024.0))

                vcodec->bit_rate = num_val * 1024.0;

        }

    }



    return 0;

}
