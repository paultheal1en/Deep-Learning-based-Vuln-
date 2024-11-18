static void usage(const char *name)

{

    (printf) (

"Usage: %s [OPTIONS] FILE\n"

"QEMU Disk Network Block Device Server\n"

"\n"

"  -h, --help                display this help and exit\n"

"  -V, --version             output version information and exit\n"

"\n"

"Connection properties:\n"

"  -p, --port=PORT           port to listen on (default `%d')\n"

"  -b, --bind=IFACE          interface to bind to (default `0.0.0.0')\n"

"  -k, --socket=PATH         path to the unix socket\n"

"                            (default '"SOCKET_PATH"')\n"

"  -e, --shared=NUM          device can be shared by NUM clients (default '1')\n"

"  -t, --persistent          don't exit on the last connection\n"

"  -v, --verbose             display extra debugging information\n"

"\n"

"Exposing part of the image:\n"

"  -o, --offset=OFFSET       offset into the image\n"

"  -P, --partition=NUM       only expose partition NUM\n"

"\n"

#ifdef __linux__

"Kernel NBD client support:\n"

"  -c, --connect=DEV         connect FILE to the local NBD device DEV\n"

"  -d, --disconnect          disconnect the specified device\n"

"\n"

#endif

"\n"

"Block device options:\n"

"  -f, --format=FORMAT       set image format (raw, qcow2, ...)\n"

"  -r, --read-only           export read-only\n"

"  -s, --snapshot            use FILE as an external snapshot, create a temporary\n"

"                            file with backing_file=FILE, redirect the write to\n"

"                            the temporary one\n"

"  -l, --load-snapshot=SNAPSHOT_PARAM\n"

"                            load an internal snapshot inside FILE and export it\n"

"                            as an read-only device, SNAPSHOT_PARAM format is\n"

"                            'snapshot.id=[ID],snapshot.name=[NAME]', or\n"

"                            '[ID_OR_NAME]'\n"

"  -n, --nocache             disable host cache\n"

"      --cache=MODE          set cache mode (none, writeback, ...)\n"

#ifdef CONFIG_LINUX_AIO

"      --aio=MODE            set AIO mode (native or threads)\n"

#endif

"      --discard=MODE        set discard mode (ignore, unmap)\n"

"      --detect-zeroes=MODE  set detect-zeroes mode (off, on, unmap)\n"

"\n"

"Report bugs to <qemu-devel@nongnu.org>\n"

    , name, NBD_DEFAULT_PORT, "DEVICE");

}
