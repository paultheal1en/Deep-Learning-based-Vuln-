static int read_dialogue(ASSContext *ass, AVBPrint *dst, const uint8_t *p,

                         int64_t *start, int *duration)

{

    int pos;

    int64_t end;

    int hh1, mm1, ss1, ms1;

    int hh2, mm2, ss2, ms2;



    if (sscanf(p, "Dialogue: %*[^,],%d:%d:%d%*c%d,%d:%d:%d%*c%d,%n",

               &hh1, &mm1, &ss1, &ms1,

               &hh2, &mm2, &ss2, &ms2, &pos) >= 8) {



        /* This is not part of the sscanf itself in order to handle an actual

         * number (which would be the Layer) or the form "Marked=N" (which is

         * the old SSA field, now replaced by Layer, and will be lead to Layer

         * being 0 here). */

        const int layer = atoi(p + 10);



        end    = (hh2*3600LL + mm2*60LL + ss2) * 100LL + ms2;

        *start = (hh1*3600LL + mm1*60LL + ss1) * 100LL + ms1;

        *duration = end - *start;



        av_bprint_clear(dst);

        av_bprintf(dst, "%u,%d,%s", ass->readorder++, layer, p + pos);



        /* right strip the buffer */

        while (dst->len > 0 &&

               dst->str[dst->len - 1] == '\r' ||

               dst->str[dst->len - 1] == '\n')

            dst->str[--dst->len] = 0;

        return 0;

    }

    return -1;

}
