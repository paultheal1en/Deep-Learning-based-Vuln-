int ff_split_xiph_headers(uint8_t *extradata, int extradata_size,

                          int first_header_size, uint8_t *header_start[3],

                          int header_len[3])

{

    int i, j;



    if (AV_RB16(extradata) == first_header_size) {

        for (i=0; i<3; i++) {

            header_len[i] = AV_RB16(extradata);

            extradata += 2;

            header_start[i] = extradata;

            extradata += header_len[i];

        }

    } else if (extradata[0] == 2) {

        for (i=0,j=1; i<2; i++,j++) {

            header_len[i] = 0;

            for (; j<extradata_size && extradata[j]==0xff; j++) {

                header_len[i] += 0xff;

            }

            if (j >= extradata_size)

                return -1;



            header_len[i] += extradata[j];

        }

        header_len[2] = extradata_size - header_len[0] - header_len[1] - j;

        extradata += j;

        header_start[0] = extradata;

        header_start[1] = header_start[0] + header_len[0];

        header_start[2] = header_start[1] + header_len[1];

    } else {

        return -1;

    }

    return 0;

}
