static int put_packetheader(NUTContext *nut, ByteIOContext *bc, int max_size, int calculate_checksum)

{

    put_flush_packet(bc);

    nut->packet_start[2]= url_ftell(bc) - 8;

    nut->written_packet_size = max_size;

    

    if(calculate_checksum)

        init_checksum(bc, update_adler32, 0);



    /* packet header */

    put_v(bc, nut->written_packet_size); /* forward ptr */



    return 0;

}
