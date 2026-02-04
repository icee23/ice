#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <memory.h>
#include <float.h>

#define fs              16000
#define window_size     26                        // for band-pass filter

#define space_s         0.005

#define max_amp         32768                     // for normalizing wave

#define PI 3.1415926535897932384626433832795028841971

#define SAFEFREE(ptr) do { \
    if ((ptr) != NULL) { \
        free(ptr); \
        (ptr) = NULL; \
    } \
} while(0)

// global variable
float alias         = fs/2;
float window_size_2 = 0.025*fs;                   // for 40Hz low-pass filter
int   window_size_r = 0.025*fs;                   // for raising
float space         = fs*space_s;
int   space_hf      = fs*space_s / 2;
double w1 = 1;
double w2 = -100; //-105; // -100;
float duration_range = 0.01;

int cand_index_len_constant = 4;

// define for test
int scase1          = 0; // 90 190
int scase2          = 0;

// don't touch this : define ELEMENT_COUNT(X) (sizeof(X) / sizeof((X)[0]))

typedef struct _complex_{//複數型別
  double real;        //實部
  double imag;        //虛部
} complex;

typedef struct _WaveHeader{
    // Riff Wave Header
    char chunkId[4];
    int  chunkSize;
    char format[4];

    // Format Subchunk
    char subChunk1Id[4];
    int  subChunk1Size;
    short  audioFormat;
    short  numChannels;
    int sampleRate;
    int byteRate;
    short  blockAlign;
    short  bitsPerSample;
    //short int extraParamSize;

    // Data Subchunk
    char subChunk2Id[4];
    int  subChunk2Size;

} WaveHeader;

typedef struct _mul_item {
	float bt;// begin time of phoneme
	float et;// ending time of phoneme
	char phoneme[16];// phoneme
	int stress;// stress
	char word[128];// word
	int posnum;// number of pos
	char pos[3][16];// pos
	int prepmnum;// number of previous PMs
	char prepm[3][16];// previous PMs
	int folpmnum;// number of following PMs
	char folpm[3][16];// following PMs
} mul_item;

typedef struct _mul_file {
	char fn[256];// file name
	char title[256];// file title
	int size;// number of mul_item
	mul_item *item;
} mul_file;

typedef struct _mul {
	int size;
	//mul_file *item;
	mul_item *item;
} mul;

typedef struct _fn_list{
	int size;
	char **fn;
} fn_list;

typedef struct _phoneme_table_item{
    char phoneme[10];
    int uv_sign;
} phoneme_table_item;

typedef struct _phoneme_table{
    int size;
    phoneme_table_item *item;
} phoneme_table;

typedef struct _fixed_seg_item{
    double *time_sq;
    int *index_sq;
    double *d_KL_sq;
    int stop_point_sq;
    int len;
    int *peak_index_sq;
    int peak_index_len;
    int cand_index[5];
    int cand_index_len;
} fixed_seg_item;

typedef struct _fixed_seg{
    int size;
    fixed_seg_item *item;
} fixed_seg;

typedef struct _choice_item{
    double time;
    int index;
    double score;
    int V_path[2];
} choice_item;

typedef struct _choice{
    int size;
    choice_item *item;
    int AV_path[2];
} choice;

void choice_free(choice *choice_, int len){
    int i;
    for(i = 0; i < len; i++){
        choice_[i].size = 0;
        choice_[i].AV_path[0] = 0;
        choice_[i].AV_path[1] = 0;
        SAFEFREE(choice_[i].item);
    }
    SAFEFREE(choice_);
}

void mul_free(mul *mul_){
	SAFEFREE(mul_->item);
	mul_->size = 0;
}

void fixed_seg_free(fixed_seg *fixed_seg_){
	int i;
	for(i = 0; i < fixed_seg_->size; i++){
        SAFEFREE(fixed_seg_->item[i].d_KL_sq);
        SAFEFREE(fixed_seg_->item[i].index_sq);
        SAFEFREE(fixed_seg_->item[i].peak_index_sq);
        SAFEFREE(fixed_seg_->item[i].time_sq);
	}
	SAFEFREE(fixed_seg_->item);
	fixed_seg_->size = 0;
}

void read_phoneme_table(char *fn, phoneme_table *phtable_){
    FILE *fp;
    char line[1024], tk[32][256], *token;
    int temp_size=0, i, tk_size;
    fp = fopen(fn, "r");
    if (!fp) {
		printf("Cannot open %s\n", fn);
		printf("read_phoneme_table error");
		exit(1);
	}
	while(!feof(fp)){
        fgets(line, 1023, fp);
		temp_size++;
	}
	temp_size--;
	phtable_->size = temp_size;
	//phtable_ = (phoneme_table *)calloc(temp_size, sizeof(phoneme_table));
	phtable_->item = (phoneme_table_item*)malloc((phtable_->size+2)*sizeof(phoneme_table_item));
	rewind(fp);
	for(i = 0; i < phtable_->size; i++){
        fgets(line, 1023, fp);
        //printf("%s", line); //---------------------------------
		tk_size = 0;
		token = strtok(line, " \t\n");
		while (token) {
			strcpy(tk[tk_size], token);
			//printf("%s\n", token); //----------------------------
			//system("pause"); //----------------------------
			tk_size++;
			token = strtok(NULL, " \t\n");
		}
		if(tk_size == 2){
            strcpy(phtable_->item[i].phoneme, tk[0]);
            if(strcmp(tk[1], "unvoiced") == 0)
                phtable_->item[i].uv_sign = 0;
            else if(strcmp(tk[1], "voiced") == 0)
                phtable_->item[i].uv_sign = 1;
		}
		else{
            printf("Wrong phoneme table");
            exit(1);
		}
	}
    fclose(fp);

    strcpy(phtable_->item[i].phoneme, "sp");
    phtable_->item[i++].uv_sign = 2;
    strcpy(phtable_->item[i].phoneme, "sil");
    phtable_->item[i].uv_sign = 3;
    phtable_->size += 2;

    /*for(i = 0; i < temp_size; i++){
        printf("%s %d\n", phtable_[i].phoneme, phtable_[i].uv_sign);
        system("pause");
	}*/
}

int load_fn_list(char *fn, fn_list *pscp){
	FILE *fp = fopen(fn, "r");
	char line[1024];
	int i;
	if (!fp) {
		fprintf(stderr, "Cannot open %s\n", fn);
		return 0;
	}

	pscp->size = 0;
	while (fscanf(fp, "%s\n", line) == 1) {
		pscp->size++;
	}
	pscp->fn = (char **)calloc(pscp->size, sizeof(char*));
	rewind(fp);
	for (i = 0; i < pscp->size; i++) {
		fscanf(fp, "%s\n", line);
		pscp->fn[i] = (char *)calloc(strlen(line) + 1, sizeof(char));
		strcpy(pscp->fn[i], line);
	}
	fclose(fp);
	return 1;
}

void lab_list(char *s1, char *pure_name_, char *path_){
    int tk_size = 0;
    int a, i;
    char *token;
    char tk[32][256];
    char s2[512] = {""};
    char temp_path[512] = {""};
    strcpy(s2, s1);
    token = strtok(s2, "./\n");
    while (token) {
        strcpy(tk[tk_size], token);
        /*printf("%d %s\n", tk_size, token);
        system("pause");*/
        tk_size++;
        token = strtok(NULL, "./\n");
    }
    //printf("%s %d\n", tk[tk_size-2], strlen(tk[tk_size-2]));
    strcpy(pure_name_, tk[tk_size-2]);
    //printf("%s\n", pure_name_);

    for(i = 0; i < tk_size-3; i++){
        strcat(temp_path, tk[i]);
        strcat(temp_path, "/");
    }
    strcpy(path_, temp_path);

    //free(token);
    for(i = 0; i < 512; i++){
        temp_path[i] = '\0';
    }
}

void mul_load_fn(char *fn, mul *pmul){
	FILE *fp = fopen(fn, "r");
	char header[1024];
	char line[1024];
	char now_fn[1024];
	char last_fn[1024];
	char tmp[256];
	int fidx;//file index
	int tmpitemsize;
	int ret;
	int i, j, k, m;
	int ofs;
	char *token;
	char tk[32][256];
	char tk_size;

	char line_temp[1000];

    //printf("%s\n", fn);

	if (!fp) {
		printf("Cannot open %s\n", fn);
		exit(1);
	}

	pmul->size = 0;

	// count number of file
	while (!feof(fp)) {
		fgets(line, 1023, fp);
		pmul->size++;
	}
	pmul->size--;

	//printf("pmul->size = %d\n", pmul->size);
	pmul->item = (mul_item *)malloc(pmul->size*sizeof(mul_item));
	rewind(fp);

	for (i = 0; i < pmul->size; i++) {
		fgets(line, 1023, fp);
		//printf("%s ", line);

		strcpy(line_temp, line);
		tk_size = 0;
		token = strtok(line, " \t\n");
		while (token) {
			strcpy(tk[tk_size], token);
			tk_size++;
/*printf("%s ", token);
system("pause");*/
			token = strtok(NULL, " \t\n");
		}

		/*for(j = 0; j < tk_size; j++){
            printf("%d %d %s %d", i, j, tk[j], tk_size);
            system("pause");
		}*/

		if (tk_size == 3) {
			pmul->item[i].bt = atof(tk[0]);
			pmul->item[i].et = atof(tk[1]);
			strcpy(pmul->item[i].phoneme, tk[2]);
			//pmul->item[i].item[j].stress = -1;
			//pmul->item[i].item[j].word[0] = "";
			//strcpy(pmul->item[i].word, "");
			pmul->item[i].word[0] = '\0';
			pmul->item[i].posnum = 0;
			pmul->item[i].prepmnum = 0;
			pmul->item[i].folpmnum = 0;

			//printf("%f %f %s\n", pmul->item[i].bt, pmul->item[i].et, pmul->item[i].phoneme);
		}
		else if (tk_size == 4) {
            //printf("%s %s %s %s\n", tk[0], tk[1], tk[2], tk[3]);
			pmul->item[i].bt = atof(tk[0]);

			pmul->item[i].et = atof(tk[1]);

			strcpy(pmul->item[i].phoneme, tk[2]);
			//pmul->item[i].item[j].stress = atoi(tk[3]);
			strcpy(pmul->item[i].word, tk[3]);
			pmul->item[i].posnum = 0;
			pmul->item[i].prepmnum = 0;
			pmul->item[i].folpmnum = 0;
			//printf("%f %f %s %s\n", pmul->item[i].bt, pmul->item[i].et, pmul->item[i].phoneme, pmul->item[i].word);
		}
		else{
			printf("tk_size > 4\n");
			printf("%s\n", line_temp);
			exit(1);
		}

        for(k = 0; k < 32; k++){
            for(m = 0; m < 256; m++){
                tk[k][m] = '\0';
            }
        }


	}

	fclose(fp);

}

short* read_wav(char *fn, WaveHeader *wavheader_){
    int i, numSamples;
    short *x;
    FILE *wavfile;
    wavfile=fopen(fn,"rb");
    if (!wavfile) {
		printf("Cannot open %s\n", fn);
		exit(1);
	}
    fread(&wavheader_, 44, 1, wavfile);
    numSamples=(wavheader_->subChunk2Size*8)/wavheader_->numChannels/wavheader_->bitsPerSample; //總共有幾個sample
    //printf("numSamples = %d\n", numSamples);
    x = (short*)malloc(sizeof(short)*numSamples); //分配它應有的記憶體
    for(i = 0; i < numSamples; i++){
        fread(&x[i], sizeof(short) , 1, wavfile);
        if(feof(wavfile))
            break;
    }
    //printf("%d\n", wavheader_->sampleRate);
    fclose(wavfile);
    return x;
}

short* read_raw(char *fn, int *numSamples_2){
    int i, numSamples_ = 0, j;
    short *x_, *temp_x;
    FILE *wavfile;
    //printf("fn = %s\n", fn);
    wavfile=fopen(fn,"rb");
    if (!wavfile) {
		printf("Cannot open %s\n", fn);
		exit(1);
	}
	while (!feof(wavfile)) {
		temp_x = (short*)malloc(1*sizeof(short));

		fread(temp_x, sizeof(short) , 1, wavfile);

		free(temp_x); temp_x = NULL;
		numSamples_++;
	}
	numSamples_--;
	//printf("numSamples_ = %d\n", numSamples_);
    x_ = (short*)malloc(sizeof(short)*numSamples_);
    rewind(wavfile);
    for(i = 0; i < numSamples_; i++){
        fread(&x_[i], sizeof(short) , 1, wavfile);
        /*printf("%d ", x[i]);
		system("pause");*/
        if(feof(wavfile))
            break;
    }
    fclose(wavfile);
    *numSamples_2 = numSamples_;
    /*printf("numSamples_ = %d\n", numSamples_);
    printf("x_raw = ");
    for(j = 0; j < numSamples_; j++){
        printf("%d-%d ", j, x[j]);
        system("pause");
    }
    printf("\n");*/
    /*printf("x_raw = ");
    for(j = scase1; j < scase2; j++){
        printf("%d ", x_[j]);
        //system("pause");
    }
    printf("\n");*/
    return x_;
}

double* low_pass_filter(int f1, int window_size_){
    int i, j;
    int start, stop, len;
    double x2 = f1/alias;
    start = (int)(-1*window_size_/2);
    stop = (int)(window_size_/2);
//printf("start=%d stop=%d\n", start, stop);
    len = window_size_;
    double *sinc_x = (double*)malloc(sizeof(double)*len);
//printf("M_PI = %f\n", M_PI);
//printf("sinc_x = ");
    j = 0;
    for(i = start; i < stop; i++){
//printf("%d %f ", f1, alias);
//printf("%lf %f %f ", x2, sin(M_PI*i*x2), (M_PI*i*x2));
        if(i == 0)
            sinc_x[j] = x2;
        else
            sinc_x[j] = x2 * sin(M_PI*i*x2) / (M_PI*i*x2);

//printf("%d %lf ", j, sinc_x[j]);
        j++;
    }
//printf("\n");
//return 1;
    return sinc_x;
}

double* filter_bank(int f1, int f2){
    int i, len;
    double *filter_low_f1, *filter_low_f2;
    filter_low_f1 = low_pass_filter(f1, window_size);
    filter_low_f2 = low_pass_filter(f2, window_size);
    len = window_size;
    //printf("len = %d\n", len);
    double *out = (double*)malloc(sizeof(double)*len);
    for(i = 0; i < len; i++){
        out[i] = filter_low_f2[i] - filter_low_f1[i];
    }
    free(filter_low_f1); filter_low_f1 = NULL;
    free(filter_low_f2); filter_low_f2 = NULL;
    return out;
}

double* hamming_window(int len){
    int i;
    double *out2 = (double*)calloc(len, sizeof(double));
    for (i = 0; i < len; i++){
        out2[i] = 0.54 - 0.46*cos(2*M_PI*i/len);
    }
    return out2;
}

float* hamming_window_float(int len){
    int i;
    float *out2 = (float*)calloc(len, sizeof(float));
    for (i = 0; i < len; i++){
        out2[i] = 0.54 - 0.46*cos(2*M_PI*i/len);
    }
    return out2;
}

void array_sum(double *arr1, const double *arr2, int window_size_){
    int len1 = window_size_;
    int i;
    for(i = 0; i < len1; i++){
        arr1[i] = arr1[i] + arr2[i];
    }
    //return arr1;
}

double* array_dot(double *arr1, const double *arr2, int window_size_){
    int len1 = window_size_;
    int i;
    double *arr14 = (double*)calloc(len1, sizeof(double));
    for(i = 0; i < len1; i++){
        arr14[i] = arr1[i] * arr2[i];
    }
    free(arr1); arr1 = NULL;
    return arr14;
}

double* array_divide(const double *arr1, const double *arr2, int window_size_){
    int len1 = window_size_;
    int i;
    double *out4 = (double*)calloc(len1, sizeof(double));
    for(i = 0; i < len1; i++){
        out4[i] = arr1[i] / arr2[i];
    }
    return out4;
}

double* array_log(const double *arr1, int window_size_){
    int len1 = window_size_;
    int i;
    double *out44 = (double*)calloc(len1, sizeof(double));
    for(i = 0; i < len1; i++){
        out44[i] = log(arr1[i]);
    }
    return out44;
}

double* convolve(const double *Signal/* SignalLen */, size_t SignalLen,
              const double *Kernel/* KernelLen */, size_t KernelLen, char *type_){
    int i, j, i1;
	double tmp;
    size_t n, len;
    double *Result = (double*)calloc((SignalLen + KernelLen - 1), sizeof(double));
//printf("fixpoint 1\n");

// check Signal, Kernel in this function


    //printf("Signal = ");
    /*for(j = scase1; j < scase2; j++){
        printf("%lf ", Signal[j]);
    }
    printf("\n");*/

    for (i=0; i<SignalLen + KernelLen - 1; i++){
		i1 = i;
		tmp = 0.0;
		for (j=0; j<KernelLen; j++){
			if(i1>=0 && i1<SignalLen)
				tmp = tmp + (Signal[i1]*Kernel[j]);

			i1 = i1-1;
			Result[i] = tmp;

		}
	}

//printf("fixpoint 2\n");
    int start, stop, center, side;
    double *temp_Result;
    if(strcmp(type_, "same") == 0){
        len = SignalLen;
        /*center = round((double)(SignalLen+KernelLen-1) / 2) - 1;
        side = floor((double)(len) / 2);
        start = center - side;
        stop = center + side;*/
        if(SignalLen > KernelLen){
            start = (int)(KernelLen/2);
            stop = start + SignalLen - 1;
        }
        else{
            printf("convolve error : SignalLen > KernelLen\n");
            exit(1);
        }
    }
    else if(strcmp(type_, "full") == 0){
        len = SignalLen+KernelLen-1;
        start = 0;
        stop = len - 1;
    }
    else{
        printf("type_ error\n");
        exit(1);
    }
    //printf("\nstart=%d stop=%d side=%d\n", start, stop, side);
    //printf("%d %d\n", SignalLen, KernelLen);
    temp_Result = (double*)calloc(len, sizeof(double));
    j = 0;
    for(i = start; i <= stop; i++){
        temp_Result[j++] = Result[i];
    }
    SAFEFREE(Result);

    /*printf("temp_Result = ");
    for(j = 0; j < len; j++){
        printf("%lf ", temp_Result[j]);
    }
    printf("\n");*/

    return temp_Result;
}

double* rectified(double *x_4, int len){
    int i;
    double *x_3 = (double*)calloc(len, sizeof(double));
    for(i = 0; i < len; i++){
        if(x_4[i] <= 0.0)
            x_3[i] = 0.00001; // 0.000001
        else
            x_3[i] = x_4[i];
    }
    SAFEFREE(x_4);
    return x_3;
}

double* DKL_100(double *Ex1_, int len){
    int i, space2;
//printf(" c0-");
    double *Ex1_log = array_log(Ex1_, len);
//printf(" c1-");
    double *out58 = (double*)calloc(len, sizeof(double));
//printf(" c2-");
    space2 = (int)(space);
//printf(" c3-");
    /*for(i = 0; i < len; i++){
        if((i+space2)>=len || (i-space2)<0){
            if((i+1) < len){
                out58[i] = 0.5*(Ex1_[i]*Ex1_log[i] - Ex1_[i]*Ex1_log[i+1]) + 0.5*(Ex1_[i+1]*Ex1_log[i+1] - Ex1_[i+1]*Ex1_log[i]);
                //out[i] = 0.5*(Ex[i]*log_Ex[i] - Ex[i]*log_Ex[i+1]) + 0.5*(Ex[i+1]*log_Ex[i+1] - Ex[i+1]*log_Ex[i])
            }
            else{
                out58[i] = 0.0;
            }
        }
        else{
            out58[i] = 0.5*(Ex1_[i-space2]*Ex1_log[i-space2] - Ex1_[i-space2]*Ex1_log[i+space2]) + 0.5*(Ex1_[i+space2]*Ex1_log[i+space2] - Ex1_[i+space2]*Ex1_log[i-space2]);
            //out[i] = 0.5*(Ex[i-space]*log_Ex[i-space] - Ex[i-space]*log_Ex[i+space]) + 0.5*(Ex[i+space]*log_Ex[i+space] - Ex[i+space]*log_Ex[i-space])
        }
    }*/
    for(i = space2; i < len-space2; i++){
        if(i-space2 < 0 || i+space2 >= len){
            printf("i=%d len=%d i-space2=%d i+space2=%d\n", i, len, i-space2, i+space2);
            exit(1);
        }
        else{
            out58[i] = 0.5*(Ex1_[i-space2]*Ex1_log[i-space2] - Ex1_[i-space2]*Ex1_log[i+space2]) + 0.5*(Ex1_[i+space2]*Ex1_log[i+space2] - Ex1_[i+space2]*Ex1_log[i-space2]);
        }
    }
    SAFEFREE(Ex1_log);
    return out58;
}

double* KL_distance(double *x1_env_, double *x2_env_, double *x3_env_,
                 double *x4_env_, double *x5_env_, double *x6_env_, int len){
    double *Ex1, *Ex2, *Ex3, *Ex4, *Ex5, *Ex6;
    double *temp_Ex1, *temp_Ex2, *temp_Ex3, *temp_Ex4, *temp_Ex5, *temp_Ex6;
    double *out85 = (double*)calloc(len, sizeof(double));
    double *XX = (double*)calloc(len, sizeof(double));
    //double *XX = (double*)malloc(sizeof(double)*len);
    //memset(XX, 0, sizeof(double)*len);
    array_sum(XX, x1_env_, len);
    array_sum(XX, x2_env_, len);
    array_sum(XX, x3_env_, len);
    array_sum(XX, x4_env_, len);
    array_sum(XX, x5_env_, len);
    array_sum(XX, x6_env_, len);
printf("check line 4 ----------------------------------------------\n");
    Ex1 = array_divide(x1_env_, XX, len);
    Ex2 = array_divide(x2_env_, XX, len);
    Ex3 = array_divide(x3_env_, XX, len);
    Ex4 = array_divide(x4_env_, XX, len);
    Ex5 = array_divide(x5_env_, XX, len);
    Ex6 = array_divide(x6_env_, XX, len);

    SAFEFREE(XX);
//printf("check line 5 ----------------------------------------------\n");
printf("check line 5 --");
    temp_Ex1 = DKL_100(Ex1, len);
    SAFEFREE(Ex1);
    array_sum(out85, temp_Ex1, len);
    SAFEFREE(temp_Ex1);
printf(" 5.1 --");

    temp_Ex2 = DKL_100(Ex2, len);
    SAFEFREE(Ex2);
    array_sum(out85, temp_Ex2, len);
    SAFEFREE(temp_Ex2);
printf(" 5.2 --");

    temp_Ex3 = DKL_100(Ex3, len);
    SAFEFREE(Ex3);
    array_sum(out85, temp_Ex3, len);
    SAFEFREE(temp_Ex3);
printf(" 5.3 --");

    temp_Ex4 = DKL_100(Ex4, len);
    SAFEFREE(Ex4);
    array_sum(out85, temp_Ex4, len);
    SAFEFREE(temp_Ex4);
printf(" 5.4 --");

    temp_Ex5 = DKL_100(Ex5, len);
    SAFEFREE(Ex5);
    array_sum(out85, temp_Ex5, len);
    SAFEFREE(temp_Ex5);
printf(" 5.5 --");

    temp_Ex6 = DKL_100(Ex6, len);
    SAFEFREE(Ex6);
    array_sum(out85, temp_Ex6, len);
    SAFEFREE(temp_Ex6);
printf(" 5.6 ----\n");

printf("check line 6 ----------------------------------------------\n");

    /*array_sum(out85, temp_Ex2, len);
    array_sum(out85, temp_Ex3, len);
    array_sum(out85, temp_Ex4, len);
    array_sum(out85, temp_Ex5, len);
    array_sum(out85, temp_Ex6, len);


    SAFEFREE(temp_Ex2);
    SAFEFREE(temp_Ex3);
    SAFEFREE(temp_Ex4);
    SAFEFREE(temp_Ex5);
    SAFEFREE(temp_Ex6);
    SAFEFREE(Ex1);
    SAFEFREE(Ex2);
    SAFEFREE(Ex3);
    SAFEFREE(Ex4);
    SAFEFREE(Ex5);
    SAFEFREE(Ex6);
    SAFEFREE(XX);*/
    return out85;
}

void remove_mul_sp(mul *mul_nosp_, mul *mulall_){
    int i, j=0;
    for(i = 0; i < mulall_->size; i++){
        if(strcmp(mulall_->item[i].phoneme, "sp") == 0 && mulall_->item[i].bt == mulall_->item[i].et){
            // pass
        }
        else{
            j++;
        }
    }
    mul_nosp_->size = j;
//printf("mul_nosp_->size  = %d\n", mul_nosp_->size);
	mul_nosp_->item = (mul_item*)malloc(mul_nosp_->size*sizeof(mul_item));
    for(j = 0, i = 0; i < mulall_->size; i++){
        if(strcmp(mulall_->item[i].phoneme, "sp") == 0 && mulall_->item[i].bt == mulall_->item[i].et){
            // pass
        }
        else{
            mul_nosp_->item[j].bt = mulall_->item[i].bt;
            mul_nosp_->item[j].et = mulall_->item[i].et;
            strcpy(mul_nosp_->item[j].phoneme, mulall_->item[i].phoneme);
            if(mulall_->item[i].word[0] != '\0'){
                strcpy(mul_nosp_->item[j].word, mulall_->item[i].word);
            }
            else{
                mul_nosp_->item[j].word[0] = '\0';
            }
            j++;
        }
    }

}

void find_candidate(mul *mul_nosp_, phoneme_table *phtable_, double *d_KL_, fixed_seg *Fixseg, int len){
    int i, j, k, phone_table_index;
    int stop_point;
    float *mul_uv = (float*)calloc(mul_nosp_->size, sizeof(float));
    double *d_KL_2;
/*for(k = 0; k < phtable_->size; k++){
    printf("%s %d\n", phtable_->item[k].phoneme, phtable_->item[k].uv_sign);
}*/
    for(i = 0; i < mul_nosp_->size; i++){
        phone_table_index = 10000;
        for(k = 0; k < phtable_->size; k++){
            if(strcmp(mul_nosp_->item[i].phoneme, phtable_->item[k].phoneme) == 0){
                mul_uv[i] = (float)phtable_->item[k].uv_sign;
                phone_table_index = k;
                break;
            }
            else {
                //printf("%d=%s %d=%s\n",i, mul_nosp_->item[i].phoneme, k, phtable_->item[k].phoneme);
            }
        }
        if(phone_table_index == 10000){
            printf("Cannot find phone_table_index\n");
            exit(1);
        }
    }
    Fixseg->size = 0;
    int point_a, point_b, m;
    for(i = 0; i < mul_nosp_->size-1; i++){
        if(mul_uv[i] == 0){
            if(mul_uv[i+1] == 1)
                Fixseg->size++;
        }
        else if(mul_uv[i] == 1){
            if(mul_uv[i+1] == 0)
                Fixseg->size++;
        }
    }
/*FILE *fp;
fp = fopen("uv_sq.txt","w");
for(j = 0; j < mul_nosp_->size; j++){
    fprintf(fp, "%f\n", mul_uv[j]);
}
fclose(fp);*/
//printf("Fixseg->size = %d\n", Fixseg->size);
    // cut segment for fixing
    Fixseg->item = (fixed_seg_item*)malloc(sizeof(fixed_seg_item)*Fixseg->size);
    for(k = 0, i = 1; i < mul_nosp_->size-1; i++){

        stop_point = (int)(round(mul_nosp_->item[i].et/10000000*fs));

        if(mul_uv[i] == 0){
            if(mul_uv[i+1] == 1){

                point_a = (int)(round(mul_nosp_->item[i-1].et/10000000*fs));
                point_b = (int)(round(mul_nosp_->item[i+1].et/10000000*fs));

                /*if((stop_point-space_hf)>=0 && (stop_point+space_hf)<mul_nosp_->size){
                    point_a = stop_point - space_hf;
                    point_b = stop_point + space_hf;
                }
                else if((stop_point-space_hf)<0){
                    point_a = 0;
                    point_b = stop_point + space_hf;
                }
                else if((stop_point+space_hf)>=mul_nosp_->size){
                    point_a = stop_point - space_hf;
                    point_b = mul_nosp_->size - 1;
                }*/

                Fixseg->item[k].len = point_b - point_a + 1;
                Fixseg->item[k].d_KL_sq = (double*)malloc(sizeof(double)*Fixseg->item[k].len);
                Fixseg->item[k].index_sq = (int*)malloc(sizeof(int)*Fixseg->item[k].len);
                Fixseg->item[k].time_sq = (double*)malloc(sizeof(double)*Fixseg->item[k].len);
                for(m = 0, j = point_a; j <= point_b; m++, j++){
                    Fixseg->item[k].d_KL_sq[m] = d_KL_[j];
                    Fixseg->item[k].index_sq[m] = j;
                    Fixseg->item[k].time_sq[m] = (double)(j / fs);
                    Fixseg->item[k].stop_point_sq = i;
                }
                k++;
            }
        }
        else if(mul_uv[i] == 1){
            if(mul_uv[i+1] == 0){

                point_a = (int)(round(mul_nosp_->item[i-1].et/10000000*fs));
                point_b = (int)(round(mul_nosp_->item[i+1].et/10000000*fs));

                /*if((stop_point-space_hf)>=0 && (stop_point+space_hf)<mul_nosp_->size){
                    point_a = stop_point - space_hf;
                    point_b = stop_point + space_hf;
                }
                else if((stop_point-space_hf)<0){
                    point_a = 0;
                    point_b = stop_point + space_hf;
                }
                else if((stop_point+space_hf)>=mul_nosp_->size){
                    point_a = stop_point - space_hf;
                    point_b = mul_nosp_->size - 1;
                }*/

                Fixseg->item[k].len = point_b - point_a + 1;
                Fixseg->item[k].d_KL_sq = (double*)malloc(sizeof(double)*Fixseg->item[k].len);
                Fixseg->item[k].index_sq = (int*)malloc(sizeof(int)*Fixseg->item[k].len);
                Fixseg->item[k].time_sq = (double*)malloc(sizeof(double)*Fixseg->item[k].len);
                for(m = 0, j = point_a; j <= point_b; m++, j++){
                    Fixseg->item[k].d_KL_sq[m] = d_KL_[j];
                    Fixseg->item[k].index_sq[m] = j;
                    Fixseg->item[k].time_sq[m] = (double)(j / fs);
                    Fixseg->item[k].stop_point_sq = i;
                }
                k++;
            }
        }
    }
    // find peak index from segment
    for(k = 0; k < Fixseg->size; k++){
        Fixseg->item[k].peak_index_len = 0;
        for(i = 1; i < Fixseg->item[k].len-1; i++){
            if(Fixseg->item[k].d_KL_sq[i]>Fixseg->item[k].d_KL_sq[i-1] && Fixseg->item[k].d_KL_sq[i]>Fixseg->item[k].d_KL_sq[i+1])
                Fixseg->item[k].peak_index_len++;
        }
    }
    for(k = 0; k < Fixseg->size; k++){
        Fixseg->item[k].peak_index_sq = (int*)malloc(sizeof(int)*Fixseg->item[k].peak_index_len);
        for(m = 0, i = 1; i < Fixseg->item[k].len-1; i++){
            if(Fixseg->item[k].d_KL_sq[i]>Fixseg->item[k].d_KL_sq[i-1] && Fixseg->item[k].d_KL_sq[i]>Fixseg->item[k].d_KL_sq[i+1])
                Fixseg->item[k].peak_index_sq[m++] = Fixseg->item[k].index_sq[i];
        }
    }
/*FILE *fp;
fp = fopen("stop_point_sq.txt","w");
for(j = 0; j < Fixseg->size; j++){
    fprintf(fp, "%d\n", Fixseg->item[j].stop_point_sq);
}
fclose(fp);*/


    // find top 4 d_KL as candidates
    int max_dKL_index;
    double dkl_max;
    for(k = 0; k < Fixseg->size; k++){
        // init
        Fixseg->item[k].cand_index_len = 0;
        for(i = 0; i < 5; i++)
            Fixseg->item[k].cand_index[i] = -1;
        d_KL_2 = (double*)malloc(sizeof(double)*len);
        for(i = 0; i < len; i++){ //d_KL_2 = d_KL_;
            d_KL_2[i] = d_KL_[i];
        }
        // 1 --------------------
        max_dKL_index = dkl_max = INT_MIN;
        for(i = 0; i < Fixseg->item[k].peak_index_len; i++){
            if(dkl_max <= d_KL_2[Fixseg->item[k].peak_index_sq[i]]){
                dkl_max = d_KL_2[Fixseg->item[k].peak_index_sq[i]];
                max_dKL_index = Fixseg->item[k].peak_index_sq[i];
            }
        }
        if(max_dKL_index>0){
            Fixseg->item[k].cand_index[0] = max_dKL_index;
            Fixseg->item[k].cand_index_len += 1;
            d_KL_2[max_dKL_index] = -1;
        }
        // 2 --------------------
        max_dKL_index = dkl_max = INT_MIN;
        for(i = 0; i < Fixseg->item[k].peak_index_len; i++){
            if(dkl_max <= d_KL_2[Fixseg->item[k].peak_index_sq[i]]){
                dkl_max = d_KL_2[Fixseg->item[k].peak_index_sq[i]];
                max_dKL_index = Fixseg->item[k].peak_index_sq[i];
            }
        }
        if(max_dKL_index>0){
            Fixseg->item[k].cand_index[1] = max_dKL_index;
            Fixseg->item[k].cand_index_len += 1;
            d_KL_2[max_dKL_index] = -1;
        }
        // 3 --------------------
        max_dKL_index = dkl_max = INT_MIN;
        for(i = 0; i < Fixseg->item[k].peak_index_len; i++){
            if(dkl_max <= d_KL_2[Fixseg->item[k].peak_index_sq[i]]){
                dkl_max = d_KL_2[Fixseg->item[k].peak_index_sq[i]];
                max_dKL_index = Fixseg->item[k].peak_index_sq[i];
            }
        }
        if(max_dKL_index>0){
            Fixseg->item[k].cand_index[2] = max_dKL_index;
            Fixseg->item[k].cand_index_len += 1;
            d_KL_2[max_dKL_index] = -1;
        }
        // 4 --------------------
        max_dKL_index = dkl_max = INT_MIN;
        for(i = 0; i < Fixseg->item[k].peak_index_len; i++){
            if(dkl_max <= d_KL_2[Fixseg->item[k].peak_index_sq[i]]){
                dkl_max = d_KL_2[Fixseg->item[k].peak_index_sq[i]];
                max_dKL_index = Fixseg->item[k].peak_index_sq[i];
            }
        }
        if(max_dKL_index>0){
            Fixseg->item[k].cand_index[3] = max_dKL_index;
            Fixseg->item[k].cand_index_len += 1;
            d_KL_2[max_dKL_index] = -1;
        }
        // 5 --------------------
        /*max_dKL_index = dkl_max = INT_MIN;
        for(i = 0; i < Fixseg->item[k].peak_index_len; i++){
            if(dkl_max <= d_KL_2[Fixseg->item[k].peak_index_sq[i]]){
                dkl_max = d_KL_2[Fixseg->item[k].peak_index_sq[i]];
                max_dKL_index = Fixseg->item[k].peak_index_sq[i];
            }
        }
        if(max_dKL_index>0){
            Fixseg->item[k].cand_index[4] = max_dKL_index;
            Fixseg->item[k].cand_index_len += 1;
        }*/
        SAFEFREE(d_KL_2);
//printf("%d : %d\n", k+1, Fixseg->item[k].cand_index_len);
    }
//system("pause");
}

choice* Viterbi(int w1_, int w2_, mul *mul_nosp_, fixed_seg *Fixseg, double *d_KL_){
    int i, j , k, n;
    int temp_b[2];
    double temp_a, max_num;
    double *t_lab = (double*)malloc(sizeof(double)*mul_nosp_->size);
    choice *all_choice;

    all_choice = (choice*)malloc(sizeof(choice)*mul_nosp_->size);

    // make all choice seq
    for(i = 0, j = 0; i < mul_nosp_->size; i++){
//printf("%d : ", i);
        if(i != Fixseg->item[j].stop_point_sq || j >= Fixseg->size){
            all_choice[i].size = 1;
            all_choice[i].item = (choice_item*)malloc(sizeof(choice_item)*all_choice[i].size);


            all_choice[i].item[0].time  = (double)(mul_nosp_->item[i].et) / 10000000.0;
            all_choice[i].item[0].index = (int)(round(all_choice[i].item[0].time * fs));

//printf("size=%d %lf %d\n", all_choice[i].size, all_choice[i].item[k].time, all_choice[i].item[k].index);
        }
        else if(i == Fixseg->item[j].stop_point_sq){
//printf("Fixseg->item[j].cand_index_len = %d ", Fixseg->item[j].cand_index_len);
            all_choice[i].size = Fixseg->item[j].cand_index_len + 1;
            all_choice[i].item = (choice_item*)malloc(sizeof(choice_item)*all_choice[i].size);

            for(k = 0; k < all_choice[i].size-1; k++){
                all_choice[i].item[k].index = Fixseg->item[j].cand_index[k];
                all_choice[i].item[k].time  = (double)(all_choice[i].item[k].index) / (double)(fs);
//printf("in for ");
//printf("size=%d %lf %d\n", all_choice[i].size, all_choice[i].item[k].time, all_choice[i].item[k].index);
            }
            // add htk lab ////////////////////////////////
            all_choice[i].item[k].time  = (double)(mul_nosp_->item[i].et) / 10000000.0;
            all_choice[i].item[k].index = (int)(round(all_choice[i].item[k].time * fs));
//printf("out for ");
//printf("%lf %d\n", all_choice[i].item[k].time, all_choice[i].item[k].index);

            j++;
        }

        t_lab[i] = (double)(mul_nosp_->item[i].et / 10000000);

        /*printf("%d : ", i);
        for(k = 0; k < all_choice[i].size; k++){
            printf("[%lf %d] ", all_choice[i].item[k].time, all_choice[i].item[k].index);
        }
        printf("%s %d ", mul_nosp_->item[i].phoneme, all_choice[i].size);
        printf("\n");*/

    }
//system("pause");
//printf("check line 7 ----------------------------------------------\n");
    // initialization & iteration
    all_choice[0].item[0].score = 0;
    all_choice[0].item[0].V_path[0] = 0;
    all_choice[0].item[0].V_path[1] = 0;

//printf("mul_nosp_->size = %d\n", mul_nosp_->size);

    for(n = 1; n < mul_nosp_->size; n++){
//printf("all_choice[n-1].size=%d all_choice[n].size=%d\n", all_choice[n-1].size, all_choice[n].size);
        for(i = 0; i < all_choice[n].size; i++){
            max_num = -DBL_MAX;
            for(j = 0; j < all_choice[n-1].size; j++){

                if(all_choice[n].item[i].time-all_choice[n-1].item[j].time > duration_range){
                    temp_a = w1*(double)(d_KL_[all_choice[n].item[i].index]) + w2*pow(((all_choice[n].item[i].time-all_choice[n-1].item[j].time)-(t_lab[n]-t_lab[n-1])), 2) + all_choice[n-1].item[j].score;
                }
                else{
                    temp_a = w1*(double)(d_KL_[all_choice[n].item[i].index]) + w2*10000000 + all_choice[n-1].item[j].score;
                }
                if(max_num < temp_a){
                    max_num = temp_a;
                    temp_b[0] = j; // temp_b = [j, i]
                    temp_b[1] = i;
                }
//printf("%d %d %lf %lf %d %d all_choice[n].item[i].index=%d\n", j, i, max_num, temp_a, temp_b[0], temp_b[1], all_choice[n].item[i].index);
            }
            if(max_num == -DBL_MAX){
                printf("Can't foind max_num in Viterbi %d %d %d\n", n, i, Fixseg->size);

                printf("all_choice[n-1].size=%d all_choice[n].size=%d j=%d i=%d\n", all_choice[n-1].size, all_choice[n].size, j ,i);
                printf("%lf %lf %lf\n", -DBL_MAX, temp_a, max_num);
                all_choice[n].item[i].score     = max_num;
                all_choice[n].item[i].V_path[0] = temp_b[0];
                all_choice[n].item[i].V_path[1] = temp_b[1];
                printf("%lf %d %d\n", all_choice[n].item[i].score, all_choice[n].item[i].V_path[0], all_choice[n].item[i].V_path[1]);

                exit(1);
            }
            else{
                all_choice[n].item[i].score     = max_num;
                all_choice[n].item[i].V_path[0] = temp_b[0];
                all_choice[n].item[i].V_path[1] = temp_b[1];
            }

            temp_b[0] = -1;
            temp_b[1] = -1;
        }

        /*printf("%d : ", n);
        for(i = 0; i < all_choice[n].size; i++){
            printf("[%d, %d, %lf] ", all_choice[n].item[i].V_path[0], all_choice[n].item[i].V_path[1], all_choice[n].item[i].score);
        }
        printf("\n");*/

        //system("pause");
    }
//printf("check line 8 ----------------------------------------------\n");
    // backtracking
    int temp_V[2], num_i_max;
    for(n = 0; n < mul_nosp_->size; n++){
        all_choice[n].AV_path[0] = 0;
        all_choice[n].AV_path[1] = 0;
    }
    for(n = mul_nosp_->size-1; n>0; n--){

        if(n == mul_nosp_->size-1){
            max_num = -DBL_MAX;
//printf("max_num = %lf all_choice[n].size = %d\n", max_num, all_choice[n].size);
            for(i = 0; i < all_choice[n].size; i++){
                if(max_num < all_choice[n].item[i].score){
                    max_num = all_choice[n].item[i].score;
                    temp_V[0] = all_choice[n].item[i].V_path[0];
                    temp_V[1] = all_choice[n].item[i].V_path[1];
                    num_i_max = temp_V[0]; // num_i_max = temp_V[1];
//printf("temp_V[%d %d]\n", temp_V[0], temp_V[1]);
//system("pause");
                }
            }
        }
        else{
            temp_V[0] = all_choice[n].item[num_i_max].V_path[0];
            temp_V[1] = all_choice[n].item[num_i_max].V_path[1];
            num_i_max = temp_V[0];
//printf("temp_V[%d %d]\n", temp_V[0], temp_V[1]);
        }

        all_choice[n].AV_path[0] = temp_V[0];
        all_choice[n].AV_path[1] = temp_V[1];

        //printf("%d %d %d\n", n, all_choice[n].AV_path[0], all_choice[n].AV_path[1]);
        //system("pause");
        temp_V[0] = -1;
        temp_V[1] = -1;
    }

/*FILE *fp;
fp = fopen("AV_path.txt","w");
for(j = 0; j < mul_nosp_->size; j++){
    fprintf(fp, "%d %d size=%d\n", all_choice[j].AV_path[0], all_choice[j].AV_path[1], all_choice[j].size);
}
fclose(fp);*/
    return all_choice;
}

int zeropad_len(int num1){
    int i;
    double a;
    for(i = 0; ; i++){
        a = pow(2, i);
        if(a >= num1){
            break;
        }
    }
    return (int)a;
}

// fft function ///////////////////////////////////////////////////////////////////////////////////
void conjugate_complex(int n,complex *in,complex *out){
    int i = 0;
    for(i=0;i<n;i++)
    {
        out[i].imag = -in[i].imag;
        out[i].real = in[i].real;
    }
}

void c_abs(complex *f,float *out,int n){
    int i = 0;
    float t;
    for(i=0;i<n;i++)
    {
        t = f[i].real * f[i].real + f[i].imag * f[i].imag;
        out[i] = sqrt(t);
    }
}

float c_value(complex f){
    return f.real * f.real + f.imag * f.imag;
}


void c_plus(complex a,complex *b,complex c){
    c.real = a.real + b->real;
    c.imag = a.imag + b->imag;
}

void c_sub(complex a,complex *b,complex c){
    c.real = a.real - b->real;
    c.imag = a.imag - b->imag;
}

void c_mul(complex a,complex *b,complex *c){
    c->real = a.real * b->real - a.imag * b->imag;
    c->imag = a.real * b->imag + a.imag * b->real;
}

void c_div(complex a,complex b,complex c){
    c.real = (a.real * b.real + a.imag * b.imag)/(b.real * b.real +b.imag * b.imag);
    c.imag = (a.imag * b.real - a.real * b.imag)/(b.real * b.real +b.imag * b.imag);
}

#define SWAP(a,b)  tempr=(a);(a)=(b);(b)=tempr

void Wn_i(int n,int i,complex *Wn,char flag){
    Wn->real = cos(2*PI*i/n);
    if(flag == 1)
        Wn->imag = -sin(2*PI*i/n);
    else if(flag == 0)
        Wn->imag = -sin(2*PI*i/n);
}

complex EE(complex a, complex b){
    complex c;

	c.real=a.real*b.real-a.imag*b.imag;
	c.imag=a.real*b.imag+a.imag*b.real;

	return c;
}

//傅立葉變化
void fft(int length, complex *f){ // *f
    complex t, wn;//中間變數
    int i,j,k,m,n,l,r,M;
    int la,lb,lc;
    double temp;

    complex u;

    /*----計算分解的級數M=log2(length)----*/
    for(i=length,M=1;(i=i/2)!=1;M++);
/*printf("check line 3-3-1 ------------------------------------------\n");
printf("length=%d ", length);*/




/*int length = 8;
complex *f = (complex*)malloc(length*sizeof(complex));
for(i = 0; i < length; i++){
    f[i].real = (float)i;
    f[i].imag = 0.0;
}*/

    /*----按照倒位序重新排列原訊號----*/
    for(i=0,j=0;i<=length-2;i++)
    {
        if(i<j)
        {
            t=f[j];
            f[j]=f[i];
            f[i]=t;
        }
        k=length/2;
//printf("i=%d j=%d k=%d\n", i, j, k);

        while(k<=j)
        {
            j=j-k;
            k=k/2;

            //printf("j=%d k=%d\n", j, k);
        }
        j=j+k;

    }


/*for(i = 0; i < length; i++){
    printf("%f %f\n", f[i].real, f[i].imag);
}

exit(1);*/


//printf("\ncheck line 3-3-2 ------------------------------------------\n");
    /*----FFT演算法----*/
    for(m=1;m<=M;m++)
    {
        temp = pow(2,m);
        la=(int)temp;     //la=2^m代表第m級每個分組所含節點數


        lb=la/2;        //lb代表第m級每個分組所含碟形單元數

        u.real = 1.0;
        u.imag = 0.0;
        wn.real = cos(PI / lb);
        wn.imag = -sin(PI / lb);

        //同時它也表示每個碟形單元上下節點之間的距離
        /*----碟形運算----*/
        for(l=0;l<=lb-1;l++)
        {

            for(n=l;n<=length-1;n=n+la) //遍歷每個分組，分組總數為N/la
            {
                lc=n+lb;  //n,lc分別代表一個碟形單元的上、下節點編號

                t = EE(f[lc], u);
                f[lc].real = f[n].real - t.real;
                f[lc].imag = f[n].imag - t.imag;
                f[n].real = f[n].real + t.real;
                f[n].imag = f[n].imag + t.imag;

                /*Wn_i(length,r,&wn,1);        //wn=Wnr
printf("wn=%f %f length=%d r=%d\n", wn.real, wn.imag, length, r);
                c_mul(f[lc],&wn,&t);            //t = f[lc] * wn複數運算
                c_sub(f[n],&t,(f[lc]));        //f[lc] = f[n] - f[lc] * Wnr
                c_plus(f[n],&t,(f[n]));        //f[n] = f[n] + f[lc] * Wnr*/
            }
            u = EE(u, wn);
        }
    }
}

//傅立葉逆變換
void ifft(int length,complex *f){
    int i=0;
    conjugate_complex(length,f,f);
    fft(length,f);
    conjugate_complex(length,f,f);
    for(i=0;i<length;i++)
    {
        f[i].imag = (f[i].imag)/length;
        f[i].real = (f[i].real)/length;
    }
}
///////////////////////////////////////////////////////////////////////////////////////////////////

complex* envelope_produce(const double *Signal/* SignalLen */, size_t SignalLen,
              const double *Kernel/* KernelLen */, size_t KernelLen){
    int i, j, k;
    int index, frame_size, frame_num, frame_size_2, temp_len, len;
    int SignalLen_2, KernelLen_2;
    frame_size = (int)(0.025 * fs);

    temp_len = frame_size + KernelLen - 1;
    len = zeropad_len(temp_len); // KernelLen

    complex *x_frame, *x_frame_mul;
    complex *x_complex = (complex*)calloc(SignalLen, sizeof(complex));
    complex *x_complex_out = (complex*)calloc(SignalLen, sizeof(complex));
    complex *h_complex = (complex*)calloc(len, sizeof(complex));

    complex *h_complex_ = (complex*)calloc(KernelLen, sizeof(complex));

//printf("len=%d\n", len);

    // initialize x_complex & h_complex
    for(i = 0; i < SignalLen; i++){
        x_complex[i].real = Signal[i];
        x_complex[i].imag = 0;
    }

    int a = (len - KernelLen) / 2;

    for(j = 0, i = 0; i < KernelLen; i++, j++){
        h_complex[i].real = Kernel[j];
    }

    /*for(j = KernelLen/2, i = 0; i < KernelLen; i++, j++){

        if(j == KernelLen)
            j = 0;
        if(i < KernelLen/2)
            h_complex_[i].real = Kernel[j];
        else if(i >= KernelLen/2)
            h_complex_[i].real = Kernel[j];
        //h_complex_[i].imag = 0;
    }

    for(j = 0, i = 0; i < len; i++){
        if(i<200 || i>=len-200)
            h_complex[i].real = h_complex_[j++].real;
    }*/

/*FILE *fp;
fp = fopen("h_complex_2.txt", "w");
for(j = 0; j < len; j++){
    fprintf(fp, "%f\n", h_complex[j].real);
}
fclose(fp);
exit(1);*/

/*for(i = 0; i < KernelLen; i++){
    printf("%f %f ", h_complex[i].real, h_complex[i].imag);
    printf("%lf\n", Kernel[i]);
}*/

//printf("check line 3-1 --------------------------------------------\n");
    fft(len, h_complex);

    /*fp = fopen("h_complex.txt", "w");
    for(j = 0; j < len; j++){
        //fprintf(fp, "%f %f\n", h_complex[j].real, h_complex[j].imag);
        fprintf(fp, "%f\n", h_complex[j].real);
    }
    fclose(fp);
    printf("h_complex\n");*/

//printf("check line 3-2 --------------------------------------------\n");
    // fft
    frame_size_2 = (int)(0.02 * fs);
    int frame_size_t = (int)(0.005 * fs);
    frame_num = ceil((float)SignalLen / (float)frame_size);
    for(i = 0; i < frame_num; i++){

        if(i == frame_num-1){ // i == frame_num-1
            x_frame = (complex*)calloc(len, sizeof(complex));
            x_frame_mul = (complex*)calloc(len, sizeof(complex));
            for(k = 0, j = i*frame_size; j < SignalLen; j++, k++){
                x_frame[k].real = x_complex[j].real; // * ham[k];
                x_frame[k].imag = x_complex[j].imag;
            }
            fft(len, x_frame);
            for(k = 0; k < len; k++){
                x_frame_mul[k] = EE(x_frame[k], h_complex[k]);
                //c_mul(x_frame[k], &h_complex[k], &x_frame_mul[k]);

                /*x_frame_mul[k].imag = x_frame[k].imag;
                x_frame_mul[k].real = x_frame[k].real;*/
            }
            ifft(len, x_frame_mul);
            for(k = 0, j = i*frame_size; k < len && j < SignalLen; j++, k++){
                x_complex_out[j].real += x_frame_mul[k].real;
                x_complex_out[j].imag += x_frame_mul[k].imag;
            }
            free(x_frame);
            free(x_frame_mul);
        }
        else{
            x_frame = (complex*)calloc(len, sizeof(complex));
            x_frame_mul = (complex*)calloc(len, sizeof(complex));
            for(k = 0, j = i*frame_size; j < frame_size*(i+1); j++, k++){
                x_frame[k].real = x_complex[j].real; // * ham[k];
                x_frame[k].imag = x_complex[j].imag;
            }
            fft(len, x_frame);
            for(k = 0; k < len; k++){
                x_frame_mul[k] = EE(x_frame[k], h_complex[k]);
                //c_mul(x_frame[k], &h_complex[k], &x_frame_mul[k]);

                /*x_frame_mul[k].imag = x_frame[k].imag;
                x_frame_mul[k].real = x_frame[k].real;*/
            }
            ifft(len, x_frame_mul);
            for(k = 0, j = i*frame_size; k < len && j < SignalLen; j++, k++){
                x_complex_out[j].real += x_frame_mul[k].real;
                x_complex_out[j].imag += x_frame_mul[k].imag;
            }
            free(x_frame);
            free(x_frame_mul);
        }

    }
//printf("check line 3-3 --------------------------------------------\n");
    free(h_complex);
    free(h_complex_);
    free(x_complex);

    return x_complex_out;
}

int main(int argc, char **argv){

	FILE *fp;
	mul mulall;
	mul mulall_nosp;
	mul mulout;
	WaveHeader wavheader;
	phoneme_table phtable;
	fixed_seg Fixseg;
	choice *all_choice;
	fn_list wav_list;
	int i, j, k, n;
    double *h1, *h2, *h3, *h4, *h5, *h6, *h40, *h40_2;
    int *numSamples;
    short *x;
    double *x_double;
    double *x1, *x2, *x3, *x4, *x5, *x6;
    double *x1_env, *x2_env, *x3_env, *x4_env, *x5_env, *x6_env;
    complex *x1_env_complex, *x2_env_complex, *x3_env_complex, *x4_env_complex, *x5_env_complex, *x6_env_complex;
    double *d_KL;

    char pure_name[128] = {""};
    char path[512] = {""};
    char lab_name[512] = {""};
    char wav_name[512] = {""};
    char outlab_name[512] = {""};
    char d_KL_C_name[512] = {""};
    char d_KL_C_path[1024] = {"d_KL_C/"}; // "../d_KL_C/"

    // argv /////////////////////////////////////////////////////
    char filelist[1024] = {""}; // CEMixed-udn_list.list
    char raw16kdir[1024] = {""};
    char labindir[1024] = {""};
    char laboutdir[1024] = {""};
    char pht_path[1024] = {""}; // phoneme_table.txt
      // check argv
    if(argc != 8 && argc != 11){
        printf("Recommended instructions : uvrefine filelist raw16kdir labindir laboutdir phoneme_table w_kl w_dur\n");
        printf("filelist         : list of file name\n");
        printf("raw16kdir        : 16k raw folder position\n");
        printf("labindir         : input lab folder position\n");
        printf("laboutdir        : output lab folder position\n");
        printf("phoneme_table    : phoneme table file\n");
        printf("w_kl             : the weight for KL distance\n");
        printf("w_dur            : the weight for duration\n");

        printf("scase num1 num2 : run a part of list from file num1 to file num2\n");

        exit(1);
    }
    strcpy(filelist, argv[1]);
    strcpy(raw16kdir, argv[2]);
    strcpy(labindir, argv[3]);
    strcpy(laboutdir, argv[4]);
    strcpy(pht_path, argv[5]);
    w1 = (double)atoi(argv[6]); // w_kl
    w2 = (double)atoi(argv[7]); // w_dur

    if(argc == 11){
        if(strcmp(argv[8], "scase") == 0){
            scase1 = atoi(argv[9]);
            scase2 = atoi(argv[10]);
        }
        else{
            printf("Recommended instructions : uvrefine filelist raw16kdir labindir laboutdir phoneme_table w_kl w_dur scase num1 num2\n");
            printf("filelist         : list of file name\n");
            printf("raw16kdir        : 16k raw folder position\n");
            printf("labindir         : input lab folder position\n");
            printf("laboutdir        : output lab folder position\n");
            printf("phoneme_table    : phoneme table file\n");
            printf("w_kl             : the weight for KL distance\n");
            printf("w_dur            : the weight for duration\n");

            printf("scase num1 num2 : run a part of list from file num1 to file num2\n");

            exit(1);
        }
    }

	// read phoneme table ///////////////////////////////////////
    read_phoneme_table(pht_path, &phtable);

	// read wav list ////////////////////////////////////////////
    if (!load_fn_list(filelist, &wav_list)) {
		printf("load_fn_list error");
		exit(1);
	}
    //printf("wav_list.size = %d\n", wav_list.size);

    // preprocessing for dKL ////////////////////////////////////
    double *hamm1 = hamming_window(window_size);
    double *hamm2 = hamming_window(window_size_2);

    /*fp = fopen("hamm1.txt", "w");
    for(j = 0; j < window_size; j++){
        fprintf(fp, "%lf\n", hamm1[j]);
    }
    fclose(fp);*/

    h1 = array_dot(low_pass_filter(400, window_size), hamm1, window_size);
    h2 = array_dot(filter_bank(800, 1500), hamm1, window_size);
    h3 = array_dot(filter_bank(1200, 2000), hamm1, window_size);
    h4 = array_dot(filter_bank(2000, 3500), hamm1, window_size);
    h5 = array_dot(filter_bank(3500, 5000), hamm1, window_size);
    h6 = array_dot(filter_bank(5000, 8000), hamm1, window_size);

    h40 = array_dot(low_pass_filter(40, window_size_2), hamm2, window_size_2);

    // process file by file /////////////////////////////////////
    if(scase2 == 0)
        scase2 = wav_list.size;
	for (i = scase1; i < scase2; i++) {


        numSamples = (int*)malloc(sizeof(int));

        // read lab /////////////////////////////////////////////
        strcpy(lab_name, labindir);
        j = strlen(labindir);
		if(labindir[j-1] != '/')
            strcat(lab_name, "/");
        strcat(lab_name, wav_list.fn[i]);
        strcat(lab_name, ".lab");
        //printf("%s\n", lab_name);
		mul_load_fn(lab_name, &mulall);
		/*printf("%d\n", mulall.size);
		for(j = 0; j < mulall.size; j++){
			printf("%d %f %f %s %s\n", j+1, mulall.item[j].bt, mulall.item[j].et, mulall.item[j].phoneme, mulall.item[j].word);
            system("pause");
		}*/

		// read wave ////////////////////////////////////////////
		strcpy(wav_name, raw16kdir);
		j = strlen(raw16kdir);
		if(raw16kdir[j-1] != '/')
            strcat(wav_name, "/");
        strcat(wav_name, wav_list.fn[i]);
        strcat(wav_name, ".raw");
        //printf("%s\n", wav_list.fn[i]);
        x = read_raw(wav_name, numSamples);
        //printf("numSamples = %d %d\n", *numSamples, numSamples);
        x_double = (double*)malloc(sizeof(double)*(*numSamples));
        for(j = 0; j < *numSamples; j++){
            x_double[j] = (double)(x[j]) / max_amp;
            /*if(j>=scase1 && j<scase2){
                printf("%lf, %d\n", x_double[j], x[j]);
                system("pause");
            }*/
        }

		// dKL //////////////////////////////////////////////////
        /*printf("x = ");
        for(j = scase1; j < scase2; j++){
            printf("%d ", x[j]);
        }
        printf("\nx_double = ");
        for(j = scase1; j < scase2; j++){
            printf("%lf ", x_double[j]);
        }
        printf("\n");*/
        /*double a[3] = {1,2,3}, b[3] = {0,1,0.5};
        int len_a = 3, len_b = 3;
        x1 = rectified(convolve(&a, len_a, &b, len_b, "full"), (len_a+len_b-1));
printf("x1 = ");
for(j = 0; j < (len_a+len_b-1); j++){
    printf("%lf ", x1[j]);
}
printf("\n");
return 1;*/
printf("check line 1 ----------------------------------------------\n");
        x1 = rectified(convolve(x_double, *numSamples, h1, window_size, "same"), *numSamples);
        x2 = rectified(convolve(x_double, *numSamples, h2, window_size, "same"), *numSamples);
        x3 = rectified(convolve(x_double, *numSamples, h3, window_size, "same"), *numSamples);    // this is filtered & rectified signal
        x4 = rectified(convolve(x_double, *numSamples, h4, window_size, "same"), *numSamples);
        x5 = rectified(convolve(x_double, *numSamples, h5, window_size, "same"), *numSamples);
        x6 = rectified(convolve(x_double, *numSamples, h6, window_size, "same"), *numSamples);
printf("check line 2 ----------------------------------------------\n");

        //-----------------------------------
        fp = fopen("filter1.txt", "w");
        for(j = 0; j < *numSamples; j++){
            fprintf(fp, "%lf\n", x1[j]);
        }
        fclose(fp);/*
        //-----------------------------------
        fp = fopen("filter2.txt", "w");
        for(j = 0; j < *numSamples; j++){
            fprintf(fp, "%lf\n", x2[j]);
        }
        fclose(fp);
        //-----------------------------------
        fp = fopen("filter3.txt", "w");
        for(j = 0; j < *numSamples; j++){
            fprintf(fp, "%lf\n", x3[j]);
        }
        fclose(fp);
        //-----------------------------------
        fp = fopen("filter4.txt", "w");
        for(j = 0; j < *numSamples; j++){
            fprintf(fp, "%lf\n", x4[j]);
        }
        fclose(fp);
        //-----------------------------------
        fp = fopen("filter5.txt", "w");
        for(j = 0; j < *numSamples; j++){
            fprintf(fp, "%lf\n", x5[j]);
        }
        fclose(fp);
        //-----------------------------------
        fp = fopen("filter6.txt", "w");
        for(j = 0; j < *numSamples; j++){
            fprintf(fp, "%lf\n", x6[j]);
        }
        fclose(fp);*/


        /*printf("x1 = ");
        for(j = scase1; j < scase2; j++){
            printf("%lf ", x1[j]);
        }
        printf("\n");*/

        // fft low-pass filter //////////////////////////////////
        x1_env_complex = envelope_produce(x1, *numSamples, h40, window_size_2);
        x2_env_complex = envelope_produce(x2, *numSamples, h40, window_size_2);
        x3_env_complex = envelope_produce(x3, *numSamples, h40, window_size_2);
        x4_env_complex = envelope_produce(x4, *numSamples, h40, window_size_2);
        x5_env_complex = envelope_produce(x5, *numSamples, h40, window_size_2);
        x6_env_complex = envelope_produce(x6, *numSamples, h40, window_size_2);

        /*fp = fopen("filter1_env_complex.txt", "w");
        for(j = 0; j < *numSamples; j++){
            //fprintf(fp, "%f %f\n", x1_env_complex[j].real, x1_env_complex[j].imag);
            fprintf(fp, "%lf\n", x1_env_complex[j].real);
        }
        fclose(fp);

        fp = fopen("filter2_env_complex.txt", "w");
        for(j = 0; j < *numSamples; j++){
            fprintf(fp, "%lf\n", x2_env_complex[j].real);
        }
        fclose(fp);

        fp = fopen("filter3_env_complex.txt", "w");
        for(j = 0; j < *numSamples; j++){
            fprintf(fp, "%lf\n", x3_env_complex[j].real);
        }
        fclose(fp);

        fp = fopen("filter4_env_complex.txt", "w");
        for(j = 0; j < *numSamples; j++){
            fprintf(fp, "%lf\n", x4_env_complex[j].real);
        }
        fclose(fp);

        fp = fopen("filter5_env_complex.txt", "w");
        for(j = 0; j < *numSamples; j++){
            fprintf(fp, "%lf\n", x5_env_complex[j].real);
        }
        fclose(fp);

        fp = fopen("filter6_env_complex.txt", "w");
        for(j = 0; j < *numSamples; j++){
            fprintf(fp, "%lf\n", x6_env_complex[j].real);
        }
        fclose(fp);*/



printf("check line 3 ----------------------------------------------\n");
        /*x1_env = convolve(x1, *numSamples, h40, window_size_2, "same");
        x2_env = convolve(x2, *numSamples, h40, window_size_2, "same");
        x3_env = convolve(x3, *numSamples, h40, window_size_2, "same");         // this is filtered signal envelope
        x4_env = convolve(x4, *numSamples, h40, window_size_2, "same");
        x5_env = convolve(x5, *numSamples, h40, window_size_2, "same");
        x6_env = convolve(x6, *numSamples, h40, window_size_2, "same");*/

        /*fp = fopen("filter1_env.txt", "w");
        for(j = 0; j < *numSamples; j++){
            fprintf(fp, "%lf\n", x1_env[j]);
        }
        fclose(fp);*/

        x1_env = (double*)calloc(*numSamples, sizeof(double));
        x2_env = (double*)calloc(*numSamples, sizeof(double));
        x3_env = (double*)calloc(*numSamples, sizeof(double));
        x4_env = (double*)calloc(*numSamples, sizeof(double));
        x5_env = (double*)calloc(*numSamples, sizeof(double));
        x6_env = (double*)calloc(*numSamples, sizeof(double));
        k = (int)(window_size_2/2);
        for(j = 0; j < *numSamples; j++){
            if(j+k >= *numSamples)
                break;
            x1_env[j] = x1_env_complex[j+k].real;
            x2_env[j] = x2_env_complex[j+k].real;
            x3_env[j] = x3_env_complex[j+k].real;
            x4_env[j] = x4_env_complex[j+k].real;
            x5_env[j] = x5_env_complex[j+k].real;
            x6_env[j] = x6_env_complex[j+k].real;
        }

        d_KL = KL_distance(x1_env, x2_env, x3_env, x4_env, x5_env, x6_env, *numSamples);
printf(" d_KL ");

        /*strcpy(d_KL_C_name, path);
        strcat(d_KL_C_name, "d_KL_C/");
        strcat(d_KL_C_name, pure_name);
        strcat(d_KL_C_name, ".txt");
        //printf("d_KL_name = %s\n", d_KL_C_name);*/
        fp = fopen("dkl1.txt", "w");
        for(j = 0; j < *numSamples; j++){
            fprintf(fp, "%lf\n", d_KL[j]);
        }
        fclose(fp);

		// remove sp ////////////////////////////////////////////
		remove_mul_sp(&mulall_nosp, &mulall);
printf(" remove_sp ");
		/*for(j = 0; j < mulall_nosp.size; j++){
            printf("%f %f %s", mulall_nosp.item[j].bt,  mulall_nosp.item[j].et,  mulall_nosp.item[j].phoneme);
            if(mulall_nosp.item[j].phoneme[0] != NULL)
                printf(" %s \n", mulall_nosp.item[j].word);
            else
                printf("\n");
		}*/

		// find candidates //////////////////////////////////////
		find_candidate(&mulall_nosp, &phtable, d_KL, &Fixseg, *numSamples);
printf(" find_candidate ");
        /*fp = fopen("test1_Fixseg.txt", "w");

        double *djeio = (double*)calloc((*numSamples), sizeof(double));

        for(k = 0; k < Fixseg.size; k++){

            for(j = 0; j < Fixseg.item[k].cand_index_len_constant; j++){
                if(Fixseg.item[k].cand_index[j] != -1)
                    djeio[Fixseg.item[k].cand_index[j]] = 1;

            }

        }
        for(j = 0; j < *numSamples; j++){

            fprintf(fp, "%lf\n", djeio[j]);
        }
        fclose(fp);
        return 0;*/

        /*strcat(outlab_name, ".txt");

        fp = fopen(outlab_name, "w");
        for(j = 0; j < *numSamples; j++){
            fprintf(fp, "%lf\n", d_KL[j]);
        }
        fclose(fp);*/

		// Viterbi //////////////////////////////////////////////
        all_choice = Viterbi(w1, w2, &mulall_nosp, &Fixseg, d_KL);
printf(" Viterbi ");
//printf("check line 9 ----------------------------------------------\n");


		// output lab ///////////////////////////////////////////
        strcpy(outlab_name, laboutdir);
        j = strlen(laboutdir);
		if(laboutdir[j-1] != '/')
            strcat(outlab_name, "/");
        strcat(outlab_name, wav_list.fn[i]);
        strcat(outlab_name, ".lab");

        mulout.size = mulall.size;
        mulout.item = (mul_item*)malloc(mulall.size*sizeof(mul_item));
        for(k = 0; k < mulout.size; k++){
            mulout.item[k].bt = mulall.item[k].bt;
            mulout.item[k].et = mulall.item[k].et;
            strcpy(mulout.item[k].phoneme, mulall.item[k].phoneme);
            if(mulall.item[k].word[0] != '\0'){
                strcpy(mulout.item[k].word, mulall.item[k].word);
                //printf("%f %f %s %s\n", mulout.item[k].bt, mulout.item[k].et, mulout.item[k].phoneme, mulout.item[k].word);
            }
            else{
                mulout.item[k].word[0] = '\0';
                //printf("%f %f %s\n", mulout.item[k].bt, mulout.item[k].et, mulout.item[k].phoneme);
            }
        }
printf(" mulout_copy ");
/*system("pause");
printf("mulout.size = %d\n", mulout.size);
printf("check line 10 ----------------------------------------------\n");
for(k = 0; k < mulall_nosp.size; k++){
    printf("%d %d %lf\n", all_choice[k].AV_path[0], all_choice[k].AV_path[1], all_choice[k].item[all_choice[k].AV_path[1]].time*10000000);
    //printf("%d %d\n", all_choice[k].AV_path[0], all_choice[k].AV_path[1]);
}*/
//printf("mulout.size=%d ", mulout.size);
//printf("mulall_nosp.size=%d ", mulall_nosp.size);
int ret;
            for(k = 0, j = 0; j < mulout.size-1; j++){
                if(j == 0){
                    printf(" in here ");
                }
/*ret = strcmp(mulout.item[j].phoneme, "sp");
printf("ret=%d ", ret);*/

            if(strcmp(mulout.item[j].phoneme, "sp")==0 && mulall.item[j].bt == mulall.item[j].et){
                mulout.item[j].et = mulout.item[j].bt;
                mulout.item[j+1].bt = mulout.item[j].et;
            }
            else{
//printf("all_choice[k].AV_path[1]=%d all_choice[k].size=%d ", all_choice[k].AV_path[1], all_choice[k].size);
//printf("all_choice[k].item[all_choice[k].AV_path[1]].time=%lf ", all_choice[k].item[all_choice[k].AV_path[1]].time);
                mulout.item[j].et = (float)(all_choice[k].item[all_choice[k].AV_path[1]].time*10000000);
                mulout.item[j+1].bt = mulout.item[j].et;
                k++;
            }
            /*printf("%d %d %f %f %s", j, k, mulout.item[j].bt, mulout.item[j].et, mulout.item[j].phoneme);
            if(mulout.item[j].word[0] != '\0'){
                printf(" %s\n", mulout.item[j].word);
            }
            else{
                printf("\n");
            }*/
            //system("pause");
        }
printf(" mulout_finished ");
//printf("check line 11 ----------------------------------------------\n");
        fp = fopen(outlab_name, "w");
        for(j = 0; j < mulout.size; j++){
            fprintf(fp, "%d %d %s", (int)mulout.item[j].bt,  (int)mulout.item[j].et,  mulout.item[j].phoneme);
            if(mulout.item[j].word[0] != '\0')
                fprintf(fp, " %s \n", mulout.item[j].word);
            else
                fprintf(fp, "\n");
        }
        fclose(fp);

        printf(" %s\n\n", wav_list.fn[i]);
        //system("pause");

        // freeeeee /////////////////////////////////////////////
        pure_name[0] = '\0';
        path[0] = '\0';
        lab_name[0] = '\0';
        wav_name[0] = '\0';
        outlab_name[0] = '\0';
        d_KL_C_name[0] = '\0';
        SAFEFREE(x);
        SAFEFREE(x_double);
        SAFEFREE(numSamples);
        SAFEFREE(x1);
        SAFEFREE(x2);
        SAFEFREE(x3);
        SAFEFREE(x4);
        SAFEFREE(x5);
        SAFEFREE(x6);
        SAFEFREE(x1_env);
        SAFEFREE(x2_env);
        SAFEFREE(x3_env);
        SAFEFREE(x4_env);
        SAFEFREE(x5_env);
        SAFEFREE(x6_env);
        SAFEFREE(x1_env_complex);
        SAFEFREE(x2_env_complex);
        SAFEFREE(x3_env_complex);
        SAFEFREE(x4_env_complex);
        SAFEFREE(x5_env_complex);
        SAFEFREE(x6_env_complex);
        SAFEFREE(d_KL);
        choice_free(all_choice, mulall_nosp.size);
        mul_free(&mulall);
        mul_free(&mulall_nosp);
        mul_free(&mulout);
        fixed_seg_free(&Fixseg);
	}

    return 0;
}
