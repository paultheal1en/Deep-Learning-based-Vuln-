static int read_naa_id(const uint8_t *p, uint64_t *p_wwn)

{

    int i;



    if ((p[1] & 0xF) == 3) {

        /* NAA designator type */

        if (p[3] != 8) {

            return -EINVAL;

        }

        *p_wwn = ldq_be_p(p + 4);

        return 0;

    }



    if ((p[1] & 0xF) == 8) {

        /* SCSI name string designator type */

        if (p[3] < 20 || memcmp(&p[4], "naa.", 4)) {

            return -EINVAL;

        }

        if (p[3] > 20 && p[24] != ',') {

            return -EINVAL;

        }

        *p_wwn = 0;

        for (i = 8; i < 24; i++) {

            char c = toupper(p[i]);

            c -= (c >= '0' && c <= '9' ? '0' : 'A' - 10);

            *p_wwn = (*p_wwn << 4) | c;

        }

        return 0;

    }



    return -EINVAL;

}
