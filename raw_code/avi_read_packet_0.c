static int avi_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    AVIContext *avi = s->priv_data;

    ByteIOContext *pb = &s->pb;

    int n, d[8], size;

    offset_t i;

    void* dstr;



    memset(d, -1, sizeof(int)*8);

   

    if (avi->dv_demux) {

        size = dv_get_packet(avi->dv_demux, pkt);

	if (size >= 0)

	    return size;

    }

        

    for(i=url_ftell(pb); !url_feof(pb); i++) {

        int j;



	if (i >= avi->movi_end) {

	    if (avi->is_odml) {

		url_fskip(pb, avi->riff_end - i);

	        avi->riff_end = avi->movi_end = url_filesize(url_fileno(pb));

	    } else

	        break;

	}



        for(j=0; j<7; j++)

            d[j]= d[j+1];

        d[7]= get_byte(pb);

        

        size= d[4] + (d[5]<<8) + (d[6]<<16) + (d[7]<<24);

        

        //parse ix##

        n= (d[2] - '0') * 10 + (d[3] - '0');

        if(    d[2] >= '0' && d[2] <= '9'

            && d[3] >= '0' && d[3] <= '9'

            && d[0] == 'i' && d[1] == 'x'

            && n < s->nb_streams

            && i + size <= avi->movi_end){

            

            url_fskip(pb, size);

        }



	//parse JUNK

        if(d[0] == 'J' && d[1] == 'U' && d[2] == 'N' && d[3] == 'K' &&

           i + size <= avi->movi_end) {

            

            url_fskip(pb, size);

        }

        

        //parse ##dc/##wb

        n= (d[0] - '0') * 10 + (d[1] - '0');

        if(    d[0] >= '0' && d[0] <= '9'

            && d[1] >= '0' && d[1] <= '9'

            && ((d[2] == 'd' && d[3] == 'c') || 

	        (d[2] == 'w' && d[3] == 'b') || 

		(d[2] == 'd' && d[3] == 'b') ||

		(d[2] == '_' && d[3] == '_'))

            && n < s->nb_streams

            && i + size <= avi->movi_end) {

        

            av_new_packet(pkt, size);

            get_buffer(pb, pkt->data, size);

            if (size & 1) {

                get_byte(pb);

		size++;

	    }

	

	    if (avi->dv_demux) {

	        dstr = pkt->destruct;

	        size = dv_produce_packet(avi->dv_demux, pkt,

		                         pkt->data, pkt->size);

		pkt->destruct = dstr;

                pkt->flags |= PKT_FLAG_KEY;

	    } else {

                AVStream *st;

                AVIStream *ast;

                st = s->streams[n];

                ast = st->priv_data;

                

                /* XXX: how to handle B frames in avi ? */

                pkt->dts = ast->frame_offset;

//                pkt->dts += ast->start;

                if(ast->sample_size)

                    pkt->dts /= ast->sample_size;

//av_log(NULL, AV_LOG_DEBUG, "dts:%Ld offset:%d %d/%d smpl_siz:%d base:%d st:%d size:%d\n", pkt->dts, ast->frame_offset, ast->scale, ast->rate, ast->sample_size, AV_TIME_BASE, n, size);

                pkt->stream_index = n;

                /* FIXME: We really should read index for that */

                if (st->codec.codec_type == CODEC_TYPE_VIDEO) {

                    if (ast->frame_offset < ast->nb_index_entries) {

                        if (ast->index_entries[ast->frame_offset].flags & AVIIF_INDEX)

                            pkt->flags |= PKT_FLAG_KEY; 

                    } else {

                        /* if no index, better to say that all frames

                           are key frames */

                        pkt->flags |= PKT_FLAG_KEY;

                    }

                } else {

                    pkt->flags |= PKT_FLAG_KEY; 

                }

                if(ast->sample_size)

                    ast->frame_offset += pkt->size;

                else

                    ast->frame_offset++;

	    }

            return size;

        }

    }

    return -1;

}
