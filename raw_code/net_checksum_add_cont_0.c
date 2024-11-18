uint32_t net_checksum_add_cont(int len, uint8_t *buf, int seq)

{

    uint32_t sum = 0;

    int i;



    for (i = seq; i < seq + len; i++) {

        if (i & 1) {

            sum += (uint32_t)buf[i - seq];

        } else {

            sum += (uint32_t)buf[i - seq] << 8;

        }

    }

    return sum;

}
