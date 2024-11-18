uint32_t net_checksum_add(int len, uint8_t *buf)

{

    uint32_t sum = 0;

    int i;



    for (i = 0; i < len; i++) {

	if (i & 1)

	    sum += (uint32_t)buf[i];

	else

	    sum += (uint32_t)buf[i] << 8;

    }

    return sum;

}
