static int read_line(AVIOContext * pb, char* line, int bufsize)

{

    int i;

    for (i = 0; i < bufsize - 1; i++) {

        int b = avio_r8(pb);

        if (b == 0)

            break;

        if (b == '\n') {

            line[i] = '\0';

            return 0;

        }

        line[i] = b;

    }

    line[i] = '\0';

    return -1;

}
