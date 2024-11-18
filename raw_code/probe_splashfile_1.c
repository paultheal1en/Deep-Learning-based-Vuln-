static FILE *probe_splashfile(char *filename, int *file_sizep, int *file_typep)

{

    FILE *fp = NULL;

    int fop_ret;

    int file_size;

    int file_type = -1;

    unsigned char buf[2] = {0, 0};

    unsigned int filehead_value = 0;

    int bmp_bpp;



    fp = fopen(filename, "rb");

    if (fp == NULL) {

        error_report("failed to open file '%s'.", filename);



    /* check file size */

    fseek(fp, 0L, SEEK_END);

    file_size = ftell(fp);

    if (file_size < 2) {

        error_report("file size is less than 2 bytes '%s'.", filename);





    /* check magic ID */

    fseek(fp, 0L, SEEK_SET);

    fop_ret = fread(buf, 1, 2, fp);








    filehead_value = (buf[0] + (buf[1] << 8)) & 0xffff;

    if (filehead_value == 0xd8ff) {

        file_type = JPG_FILE;

    } else {

        if (filehead_value == 0x4d42) {

            file_type = BMP_FILE;



    if (file_type < 0) {

        error_report("'%s' not jpg/bmp file,head:0x%x.",

                         filename, filehead_value);





    /* check BMP bpp */

    if (file_type == BMP_FILE) {

        fseek(fp, 28, SEEK_SET);

        fop_ret = fread(buf, 1, 2, fp);

        bmp_bpp = (buf[0] + (buf[1] << 8)) & 0xffff;

        if (bmp_bpp != 24) {

            error_report("only 24bpp bmp file is supported.");






    /* return values */

    *file_sizep = file_size;

    *file_typep = file_type;

