static void help(void)

{

    printf("qemu-img version " QEMU_VERSION ", Copyright (c) 2004-2008 Fabrice Bellard\n"

           "usage: qemu-img command [command options]\n"

           "QEMU disk image utility\n"

           "\n"

           "Command syntax:\n"

           "  check [-f fmt] filename\n"

           "  create [-F fmt] [-b base_image] [-f fmt] [-o options] filename [size]\n"

           "  commit [-f fmt] filename\n"

           "  convert [-c] [-f fmt] [-O output_fmt] [-o options] [-B output_base_image] filename [filename2 [...]] output_filename\n"

           "  info [-f fmt] filename\n"

           "  snapshot [-l | -a snapshot | -c snapshot | -d snapshot] filename\n"

           "\n"

           "Command parameters:\n"

           "  'filename' is a disk image filename\n"

           "  'base_image' is the read-only disk image which is used as base for a copy on\n"

           "    write image; the copy on write image only stores the modified data\n"

           "  'output_base_image' forces the output image to be created as a copy on write\n"

           "    image of the specified base image; 'output_base_image' should have the same\n"

           "    content as the input's base image, however the path, image format, etc may\n"

           "    differ\n"

           "  'fmt' is the disk image format. It is guessed automatically in most cases\n"

           "  'size' is the disk image size in kilobytes. Optional suffixes\n"

           "    'M' (megabyte, 1024 * 1024) and 'G' (gigabyte, 1024 * 1024 * 1024) are\n"

           "    supported any 'k' or 'K' is ignored\n"

           "  'output_filename' is the destination disk image filename\n"

           "  'output_fmt' is the destination format\n"

           "  'options' is a comma separated list of format specific options in a\n"

           "    name=value format. Use -o ? for an overview of the options supported by the\n"

           "    used format\n"

           "  '-c' indicates that target image must be compressed (qcow format only)\n"

           "  '-h' with or without a command shows this help and lists the supported formats\n"

           "\n"

           "Parameters to snapshot subcommand:\n"

           "  'snapshot' is the name of the snapshot to create, apply or delete\n"

           "  '-a' applies a snapshot (revert disk to saved state)\n"

           "  '-c' creates a snapshot\n"

           "  '-d' deletes a snapshot\n"

           "  '-l' lists all snapshots in the given image\n"

           );

    printf("\nSupported formats:");

    bdrv_iterate_format(format_print, NULL);

    printf("\n");

    exit(1);

}
