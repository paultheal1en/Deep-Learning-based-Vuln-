static uint32_t read_long(const unsigned char *p)

{

	return (p[0]<<24)|(p[1]<<16)|(p[2]<<8)|p[3];

}
