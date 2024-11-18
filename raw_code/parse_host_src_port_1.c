int parse_host_src_port(struct sockaddr_in *haddr,

                        struct sockaddr_in *saddr,

                        const char *input_str)

{

    char *str = strdup(input_str);

    char *host_str = str;

    char *src_str;

    const char *src_str2;

    char *ptr;



    /*

     * Chop off any extra arguments at the end of the string which

     * would start with a comma, then fill in the src port information

     * if it was provided else use the "any address" and "any port".

     */

    if ((ptr = strchr(str,',')))

        *ptr = '\0';



    if ((src_str = strchr(input_str,'@'))) {

        *src_str = '\0';

        src_str++;

    }



    if (parse_host_port(haddr, host_str) < 0)

        goto fail;



    src_str2 = src_str;

    if (!src_str || *src_str == '\0')

        src_str2 = ":0";



    if (parse_host_port(saddr, src_str2) < 0)

        goto fail;



    free(str);

    return(0);



fail:

    free(str);

    return -1;

}
