static inline int cris_abs(int n)

{

	int r;

	asm ("abs\t%1, %0\n" : "=r" (r) : "r" (n));

	return r;

}
