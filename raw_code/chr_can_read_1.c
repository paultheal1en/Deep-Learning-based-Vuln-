static int chr_can_read(void *opaque)

{

    int can_read;

    SCLPConsole *scon = opaque;



    can_read = SIZE_BUFFER_VT220 - scon->iov_data_len;



    return can_read;

}
