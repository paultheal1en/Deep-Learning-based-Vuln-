static void set_algorythm( OPL_CH *CH)

{

	INT32 *carrier = &outd[0];

	CH->connect1 = CH->CON ? carrier : &feedback2;

	CH->connect2 = carrier;

}
