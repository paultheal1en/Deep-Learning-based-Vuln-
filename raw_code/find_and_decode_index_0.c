static int find_and_decode_index(NUTContext *nut){

    AVFormatContext *s= nut->avf;

    ByteIOContext *bc = s->pb;

    uint64_t tmp, end;

    int i, j, syncpoint_count;

    int64_t filesize= url_fsize(bc);

    int64_t *syncpoints;

    int8_t *has_keyframe;



    url_fseek(bc, filesize-12, SEEK_SET);

    url_fseek(bc, filesize-get_be64(bc), SEEK_SET);

    if(get_be64(bc) != INDEX_STARTCODE){

        av_log(s, AV_LOG_ERROR, "no index at the end\n");

        return -1;

    }



    end= get_packetheader(nut, bc, 1, INDEX_STARTCODE);

    end += url_ftell(bc);



    ff_get_v(bc); //max_pts

    GET_V(syncpoint_count, tmp < INT_MAX/8 && tmp > 0)

    syncpoints= av_malloc(sizeof(int64_t)*syncpoint_count);

    has_keyframe= av_malloc(sizeof(int8_t)*(syncpoint_count+1));

    for(i=0; i<syncpoint_count; i++){

        GET_V(syncpoints[i], tmp>0)

        if(i)

            syncpoints[i] += syncpoints[i-1];

    }



    for(i=0; i<s->nb_streams; i++){

        int64_t last_pts= -1;

        for(j=0; j<syncpoint_count;){

            uint64_t x= ff_get_v(bc);

            int type= x&1;

            int n= j;

            x>>=1;

            if(type){

                int flag= x&1;

                x>>=1;

                if(n+x >= syncpoint_count + 1){

                    av_log(s, AV_LOG_ERROR, "index overflow A\n");

                    return -1;

                }

                while(x--)

                    has_keyframe[n++]= flag;

                has_keyframe[n++]= !flag;

            }else{

                while(x != 1){

                    if(n>=syncpoint_count + 1){

                        av_log(s, AV_LOG_ERROR, "index overflow B\n");

                        return -1;

                    }

                    has_keyframe[n++]= x&1;

                    x>>=1;

                }

            }

            if(has_keyframe[0]){

                av_log(s, AV_LOG_ERROR, "keyframe before first syncpoint in index\n");

                return -1;

            }

            assert(n<=syncpoint_count+1);

            for(; j<n; j++){

                if(has_keyframe[j]){

                    uint64_t B, A= ff_get_v(bc);

                    if(!A){

                        A= ff_get_v(bc);

                        B= ff_get_v(bc);

                        //eor_pts[j][i] = last_pts + A + B

                    }else

                        B= 0;

                    av_add_index_entry(

                        s->streams[i],

                        16*syncpoints[j-1],

                        last_pts + A,

                        0,

                        0,

                        AVINDEX_KEYFRAME);

                    last_pts += A + B;

                }

            }

        }

    }



    if(skip_reserved(bc, end) || get_checksum(bc)){

        av_log(s, AV_LOG_ERROR, "index checksum mismatch\n");

        return -1;

    }

    return 0;

}
