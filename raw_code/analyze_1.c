static int analyze(const uint8_t *buf, int size, int packet_size, int *index)

{

    int stat[TS_MAX_PACKET_SIZE];

    int i;

    int x = 0;

    int best_score = 0;



    memset(stat, 0, packet_size * sizeof(int));



    for (x = i = 0; i < size - 3; i++) {

        if (buf[i] == 0x47 && !(buf[i + 1] & 0x80) && (buf[i + 3] & 0x30)) {

            stat[x]++;

            if (stat[x] > best_score) {

                best_score = stat[x];

                if (index)

                    *index = x;

            }

        }



        x++;

        if (x == packet_size)

            x = 0;

    }



    return best_score;

}
