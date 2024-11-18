static int blkdebug_open(BlockDriverState *bs, const char *filename, int flags)

{

    BDRVBlkdebugState *s = bs->opaque;

    int ret;

    char *config, *c;



    /* Parse the blkdebug: prefix */

    if (strncmp(filename, "blkdebug:", strlen("blkdebug:"))) {

        return -EINVAL;

    }

    filename += strlen("blkdebug:");



    /* Read rules from config file */

    c = strchr(filename, ':');

    if (c == NULL) {

        return -EINVAL;

    }



    config = strdup(filename);

    config[c - filename] = '\0';

    ret = read_config(s, config);

    free(config);

    if (ret < 0) {

        return ret;

    }

    filename = c + 1;



    /* Set initial state */

    s->vars.state = 1;



    /* Open the backing file */

    ret = bdrv_file_open(&bs->file, filename, flags);

    if (ret < 0) {

        return ret;

    }



    return 0;

}
