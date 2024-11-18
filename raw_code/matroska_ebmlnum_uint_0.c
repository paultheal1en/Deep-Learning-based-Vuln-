matroska_ebmlnum_uint (uint8_t  *data,

                       uint32_t  size,

                       uint64_t *num)

{

    int len_mask = 0x80, read = 1, n = 1, num_ffs = 0;

    uint64_t total;



    if (size <= 0)

        return AVERROR_INVALIDDATA;



    total = data[0];

    while (read <= 8 && !(total & len_mask)) {

        read++;

        len_mask >>= 1;

    }

    if (read > 8)

        return AVERROR_INVALIDDATA;



    if ((total &= (len_mask - 1)) == len_mask - 1)

        num_ffs++;

    if (size < read)

        return AVERROR_INVALIDDATA;

    while (n < read) {

        if (data[n] == 0xff)

            num_ffs++;

        total = (total << 8) | data[n];

        n++;

    }



    if (!total)

        return AVERROR_INVALIDDATA;



    if (read == num_ffs)

        *num = (uint64_t)-1;

    else

        *num = total;



    return read;

}
