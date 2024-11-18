static int read_braindead_odml_indx(AVFormatContext *s, int frame_num){

    AVIContext *avi = s->priv_data;

    AVIOContext *pb = s->pb;

    int longs_pre_entry= avio_rl16(pb);

    int index_sub_type = avio_r8(pb);

    int index_type     = avio_r8(pb);

    int entries_in_use = avio_rl32(pb);

    int chunk_id       = avio_rl32(pb);

    int64_t base       = avio_rl64(pb);

    int stream_id= 10*((chunk_id&0xFF) - '0') + (((chunk_id>>8)&0xFF) - '0');

    AVStream *st;

    AVIStream *ast;

    int i;

    int64_t last_pos= -1;

    int64_t filesize= avi->fsize;



    av_dlog(s, "longs_pre_entry:%d index_type:%d entries_in_use:%d chunk_id:%X base:%16"PRIX64"\n",

            longs_pre_entry,index_type, entries_in_use, chunk_id, base);



    if(stream_id >= s->nb_streams || stream_id < 0)

        return -1;

    st= s->streams[stream_id];

    ast = st->priv_data;



    if(index_sub_type)

        return -1;



    avio_rl32(pb);



    if(index_type && longs_pre_entry != 2)

        return -1;

    if(index_type>1)

        return -1;



    if(filesize > 0 && base >= filesize){

        av_log(s, AV_LOG_ERROR, "ODML index invalid\n");

        if(base>>32 == (base & 0xFFFFFFFF) && (base & 0xFFFFFFFF) < filesize && filesize <= 0xFFFFFFFF)

            base &= 0xFFFFFFFF;

        else

            return -1;

    }



    for(i=0; i<entries_in_use; i++){

        if(index_type){

            int64_t pos= avio_rl32(pb) + base - 8;

            int len    = avio_rl32(pb);

            int key= len >= 0;

            len &= 0x7FFFFFFF;



#ifdef DEBUG_SEEK

            av_log(s, AV_LOG_ERROR, "pos:%"PRId64", len:%X\n", pos, len);

#endif

            if(url_feof(pb))

                return -1;



            if(last_pos == pos || pos == base - 8)

                avi->non_interleaved= 1;

            if(last_pos != pos && (len || !ast->sample_size))

                av_add_index_entry(st, pos, ast->cum_len, len, 0, key ? AVINDEX_KEYFRAME : 0);



            ast->cum_len += get_duration(ast, len);

            last_pos= pos;

        }else{

            int64_t offset, pos;

            int duration;

            offset = avio_rl64(pb);

            avio_rl32(pb);       /* size */

            duration = avio_rl32(pb);



            if(url_feof(pb))

                return -1;



            pos = avio_tell(pb);



            if(avi->odml_depth > MAX_ODML_DEPTH){

                av_log(s, AV_LOG_ERROR, "Too deeply nested ODML indexes\n");

                return -1;

            }



            avio_seek(pb, offset+8, SEEK_SET);

            avi->odml_depth++;

            read_braindead_odml_indx(s, frame_num);

            avi->odml_depth--;

            frame_num += duration;



            avio_seek(pb, pos, SEEK_SET);

        }

    }

    avi->index_loaded=1;

    return 0;

}
