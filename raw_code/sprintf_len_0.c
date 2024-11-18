sprintf_len(char *string, const char *format, ...)

#else

sprintf_len(va_alist) va_dcl

#endif

{

	va_list args;

#ifdef __STDC__

	va_start(args, format);

#else

	char *string;

	char *format;

	va_start(args);

	string = va_arg(args, char *);

	format = va_arg(args, char *);

#endif

	vsprintf(string, format, args);

	return strlen(string);

}
