static FFServerIPAddressACL* parse_dynamic_acl(FFServerStream *stream, HTTPContext *c)

{

    FILE* f;

    char line[1024];

    char  cmd[1024];

    FFServerIPAddressACL *acl = NULL;

    int line_num = 0;

    const char *p;



    f = fopen(stream->dynamic_acl, "r");

    if (!f) {

        perror(stream->dynamic_acl);

        return NULL;

    }



    acl = av_mallocz(sizeof(FFServerIPAddressACL));



    /* Build ACL */

    for(;;) {

        if (fgets(line, sizeof(line), f) == NULL)

            break;

        line_num++;

        p = line;

        while (av_isspace(*p))

            p++;

        if (*p == '\0' || *p == '#')

            continue;

        ffserver_get_arg(cmd, sizeof(cmd), &p);



        if (!av_strcasecmp(cmd, "ACL"))

            ffserver_parse_acl_row(NULL, NULL, acl, p, stream->dynamic_acl, line_num);

    }

    fclose(f);

    return acl;

}
