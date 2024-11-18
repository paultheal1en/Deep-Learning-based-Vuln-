static int hex_to_data(uint8_t *data, const char *p)

{

    int c, len, v;



    len = 0;

    v = 1;

    for(;;) {

        skip_spaces(&p);

        if (*p == '\0')

            break;

        c = toupper((unsigned char)*p++);

        if (c >= '0' && c <= '9')

            c = c - '0';

        else if (c >= 'A' && c <= 'F')

            c = c - 'A' + 10;

        else

            break;

        v = (v << 4) | c;

        if (v & 0x100) {

            if (data)

                data[len] = v;

            len++;

            v = 1;

        }

    }

    return len;

}
