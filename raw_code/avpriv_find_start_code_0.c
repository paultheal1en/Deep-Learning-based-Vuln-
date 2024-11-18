const uint8_t *avpriv_find_start_code(const uint8_t *av_restrict p,

                                      const uint8_t *end,

                                      uint32_t *av_restrict state)

{

    int i;



    assert(p <= end);

    if (p >= end)

        return end;



    for (i = 0; i < 3; i++) {

        uint32_t tmp = *state << 8;

        *state = tmp + *(p++);

        if (tmp == 0x100 || p == end)

            return p;

    }



    while (p < end) {

        if      (p[-1] > 1      ) p += 3;

        else if (p[-2]          ) p += 2;

        else if (p[-3]|(p[-1]-1)) p++;

        else {

            p++;

            break;

        }

    }



    p = FFMIN(p, end) - 4;

    *state = AV_RB32(p);



    return p + 4;

}
