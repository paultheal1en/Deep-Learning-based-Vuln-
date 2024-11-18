static void help(int exitcode)

{

    const char *options_help =

#define QEMU_OPTIONS_GENERATE_HELP

#include "qemu-options-wrapper.h"

        ;

    version();

    printf("usage: %s [options] [disk_image]\n"

           "\n"

           "'disk_image' is a raw hard disk image for IDE hard disk 0\n"

           "\n"

           "%s\n"

           "During emulation, the following keys are useful:\n"

           "ctrl-alt-f      toggle full screen\n"

           "ctrl-alt-n      switch to virtual console 'n'\n"

           "ctrl-alt        toggle mouse and keyboard grab\n"

           "\n"

           "When using -nographic, press 'ctrl-a h' to get some help.\n",

           error_get_progname(),

           options_help);

    exit(exitcode);

}
