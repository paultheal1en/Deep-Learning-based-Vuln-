read_help(void)

{

	printf(

"\n"

" reads a range of bytes from the given offset\n"

"\n"

" Example:\n"

" 'read -v 512 1k' - dumps 1 kilobyte read from 512 bytes into the file\n"

"\n"

" Reads a segment of the currently open file, optionally dumping it to the\n"

" standard output stream (with -v option) for subsequent inspection.\n"

" -p, -- use bdrv_pread to read the file\n"

" -P, -- use a pattern to verify read data\n"

" -C, -- report statistics in a machine parsable format\n"

" -v, -- dump buffer to standard output\n"

" -q, -- quite mode, do not show I/O statistics\n"

"\n");

}
