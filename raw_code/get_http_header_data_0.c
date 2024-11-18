static int get_http_header_data(MMSHContext *mmsh)

{

    MMSContext *mms = &mmsh->mms;

    int res, len;

    ChunkType chunk_type;



    for (;;) {

        len = 0;

        res = chunk_type = get_chunk_header(mmsh, &len);

        if (res < 0) {

            return res;

        } else if (chunk_type == CHUNK_TYPE_ASF_HEADER){

            // get asf header and stored it

            if (!mms->header_parsed) {

                if (mms->asf_header) {

                    if (len != mms->asf_header_size) {

                        mms->asf_header_size = len;

                        av_dlog(NULL, "Header len changed from %d to %d\n",

                                mms->asf_header_size, len);

                        av_freep(&mms->asf_header);

                    }

                }

                mms->asf_header = av_mallocz(len);

                if (!mms->asf_header) {

                    return AVERROR(ENOMEM);

                }

                mms->asf_header_size = len;

            }

            if (len > mms->asf_header_size) {

                av_log(NULL, AV_LOG_ERROR,

                       "Asf header packet len = %d exceed the asf header buf size %d\n",

                       len, mms->asf_header_size);

                return AVERROR(EIO);

            }

            res = ffurl_read_complete(mms->mms_hd, mms->asf_header, len);

            if (res != len) {

                av_log(NULL, AV_LOG_ERROR,

                       "Recv asf header data len %d != expected len %d\n", res, len);

                return AVERROR(EIO);

            }

            mms->asf_header_size = len;

            if (!mms->header_parsed) {

                res = ff_mms_asf_header_parser(mms);

                mms->header_parsed = 1;

                return res;

            }

        } else if (chunk_type == CHUNK_TYPE_DATA) {

            // read data packet and do padding

            return read_data_packet(mmsh, len);

        } else {

            if (len) {

                if (len > sizeof(mms->in_buffer)) {

                    av_log(NULL, AV_LOG_ERROR,

                           "Other packet len = %d exceed the in_buffer size %zu\n",

                           len, sizeof(mms->in_buffer));

                    return AVERROR(EIO);

                }

                res = ffurl_read_complete(mms->mms_hd, mms->in_buffer, len);

                if (res != len) {

                    av_log(NULL, AV_LOG_ERROR, "Read other chunk type data failed!\n");

                    return AVERROR(EIO);

                } else {

                    av_dlog(NULL, "Skip chunk type %d \n", chunk_type);

                    continue;

                }

            }

        }

    }

    return 0;

}
