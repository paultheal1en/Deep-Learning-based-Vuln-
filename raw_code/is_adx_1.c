int is_adx(const unsigned char *buf,size_t bufsize)

{

	int	offset;



	if (buf[0]!=0x80) return 0;

	offset = (read_long(buf)^0x80000000)+4;

	if (bufsize<offset || memcmp(buf+offset-6,"(c)CRI",6)) return 0;

	return offset;

}
