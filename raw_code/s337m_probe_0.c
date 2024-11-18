static int s337m_probe(AVProbeData *p)

{

    uint64_t state = 0;

    int markers[3] = { 0 };

    int i, sum, max, data_type, data_size, offset;

    uint8_t *buf;



    for (buf = p->buf; buf < p->buf + p->buf_size; buf++) {

        state = (state << 8) | *buf;

        if (!IS_LE_MARKER(state))

            continue;



        if (IS_16LE_MARKER(state)) {

            data_type = AV_RL16(buf + 1);

            data_size = AV_RL16(buf + 3);

            buf += 4;

        } else {

            data_type = AV_RL24(buf + 1);

            data_size = AV_RL24(buf + 4);

            buf += 6;

        }



        if (s337m_get_offset_and_codec(NULL, state, data_type, data_size, &offset, NULL))

            continue;



        i = IS_16LE_MARKER(state) ? 0 : IS_20LE_MARKER(state) ? 1 : 2;

        markers[i]++;



        buf  += offset;

        state = 0;

    }



    sum = max = 0;

    for (i = 0; i < FF_ARRAY_ELEMS(markers); i++) {

        sum += markers[i];

        if (markers[max] < markers[i])

            max = i;

    }



    if (markers[max] > 3 && markers[max] * 4 > sum * 3)

        return AVPROBE_SCORE_EXTENSION + 1;



    return 0;

}
