putsum(uint8_t *data, uint32_t n, uint32_t sloc, uint32_t css, uint32_t cse)

{

    uint32_t sum;



    if (cse && cse < n)

        n = cse + 1;

    if (sloc < n-1) {

        sum = net_checksum_add(n-css, data+css);

        stw_be_p(data + sloc, net_checksum_finish(sum));

    }

}
