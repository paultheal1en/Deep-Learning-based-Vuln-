static int cdrom_read_toc_raw(IDEState *s, uint8_t *buf, int msf, 

                              int session_num)

{

    uint8_t *q;

    int nb_sectors, len;

    

    q = buf + 2;

    *q++ = 1; /* first session */

    *q++ = 1; /* last session */



    *q++ = 1; /* session number */

    *q++ = 0x14; /* data track */

    *q++ = 0; /* track number */

    *q++ = 0xa0; /* lead-in */

    *q++ = 0; /* min */

    *q++ = 0; /* sec */

    *q++ = 0; /* frame */

    *q++ = 0;

    *q++ = 1; /* first track */

    *q++ = 0x00; /* disk type */

    *q++ = 0x00;

    

    *q++ = 1; /* session number */

    *q++ = 0x14; /* data track */

    *q++ = 0; /* track number */

    *q++ = 0xa1;

    *q++ = 0; /* min */

    *q++ = 0; /* sec */

    *q++ = 0; /* frame */

    *q++ = 0;

    *q++ = 1; /* last track */

    *q++ = 0x00;

    *q++ = 0x00;

    

    *q++ = 1; /* session number */

    *q++ = 0x14; /* data track */

    *q++ = 0; /* track number */

    *q++ = 0xa2; /* lead-out */

    *q++ = 0; /* min */

    *q++ = 0; /* sec */

    *q++ = 0; /* frame */

    nb_sectors = s->nb_sectors >> 2;

    if (msf) {

        *q++ = 0; /* reserved */

        lba_to_msf(q, nb_sectors);

        q += 3;

    } else {

        cpu_to_ube32(q, nb_sectors);

        q += 4;

    }



    *q++ = 1; /* session number */

    *q++ = 0x14; /* ADR, control */

    *q++ = 0;    /* track number */

    *q++ = 1;    /* point */

    *q++ = 0; /* min */

    *q++ = 0; /* sec */

    *q++ = 0; /* frame */

    *q++ = 0; 

    *q++ = 0; 

    *q++ = 0; 

    *q++ = 0; 



    len = q - buf;

    cpu_to_ube16(buf, len - 2);

    return len;

}
