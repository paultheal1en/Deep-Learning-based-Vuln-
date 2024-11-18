const uint8_t *ff_h263_find_resync_marker(MpegEncContext *s, const uint8_t *av_restrict p, const uint8_t *av_restrict end)

{

    av_assert2(p < end);



    end-=2;

    p++;

    if(s->resync_marker){

        for(;p<end; p+=2){

            if(!*p){

                if     (!p[-1] && p[1]) return p - 1;

                else if(!p[ 1] && p[2]) return p;

            }

        }

    }

    return end+2;

}
