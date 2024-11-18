static int get_str(ByteIOContext *bc, char *string, int maxlen){

    int len= get_v(bc);

    

    if(len && maxlen)

        get_buffer(bc, string, FFMIN(len, maxlen));

    while(len > maxlen){

        get_byte(bc);

        len--;

    }



    if(maxlen)

        string[FFMIN(len, maxlen-1)]= 0;

    

    if(maxlen == len)

        return -1;

    else

        return 0;

}
