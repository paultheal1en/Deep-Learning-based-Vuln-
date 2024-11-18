do_cksum(uint8_t *dp, uint8_t *de)

{

    unsigned int bsum[2] = {0, 0}, i, sum;



    for (i = 1; dp < de; bsum[i^=1] += *dp++)

        ;

    sum = (bsum[0] << 8) + bsum[1];

    sum = (sum >> 16) + (sum & 0xffff);

    return ~(sum + (sum >> 16));

}
