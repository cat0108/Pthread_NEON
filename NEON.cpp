#include<iostream>
#include<pthread.h>
#include<stdio.h>
#include<arm_neon.h>
#include<sys/time.h>
#include<semaphore.h>//�ź���������ͷ�ļ�
using namespace std;
alignas(16) float gdata[10000][10000];//���ж������
float gdata2[10000][10000];
float gdata1[10000][10000];
float gdata3[10000][10000];
int n;
//�����߳����ݽṹ
typedef struct {
	int t_id;	//�̱߳��
}threadParam_t;
const int Num_thread = 8;//8��CPU

sem_t sem_leader;		//���г���������̵߳��ź���
sem_t sem_Division[Num_thread - 1];//�жϳ����Ƿ���ɵ��ź���
sem_t sem_Elimination[Num_thread - 1];//�Ƿ���Խ�����һ����ȥ���ź���
//ʹ��barrier�汾
pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;


//�̺߳���1��semaphore�汾
void* threadFunc_NEON(void* Param)//����void���͵�ָ���ָ���������͵�����
{
	threadParam_t* p = (threadParam_t*)Param;//ǿ������ת�����߳����ݽṹ
	int t_id = p->t_id;//�̵߳ı�Ż�ȡ
	//��һ���̸߳���������������������Ҳ����к�����Ԫ����
	for (int k = 0; k < n; k++)
	{
		float32x4_t r0, r1, r2, r3;//��·���㣬�����ĸ�float�����Ĵ���
		if (t_id == 0)
		{
			r0 = vmovq_n_f32(gdata[k][k]);//��ʼ��4������gdatakk������
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				r1 = vld1q_f32(gdata[k] + j);
				r1 = vdivq_f32(r1, r0);//������������λ���
				vst1q_f32(gdata[k], r1);//���������·Ż��ڴ�
			}
			//��ʣ�಻��4�������ݽ�����Ԫ
			for (j; j < n; j++)
			{
				gdata[k][j] = gdata[k][j] / gdata[k][k];
			}
			gdata[k][k] = 1.0;
		}
		else {
			sem_wait(&sem_Division[t_id - 1]);//����ֱ�������������
		}
		if (t_id == 0)//����������������������߳�
		{
			for (int i = 0; i < Num_thread - 1; i++)
			{
				sem_post(&sem_Division[i]);
			}
		}
		//���񻮷֣�����Ϊ��λ���л���

		for (int i = k + 1 + t_id; i < n; i += Num_thread)
		{
			//�����������ѭ�����SIMD����
			r0 = vmovq_n_f32(gdata[i][k]);
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				r1 = vld1q_f32(gdata[k] + j);
				r2 = vld1q_f32(gdata[i] + j);
				r3 = vmulq_f32(r0, r1);
				r2 = vsubq_f32(r2, r3);
				vst1q_f32(gdata[i] + j, r2);
			}
			for (j; j < n; j++)
			{
				gdata[i][j] = gdata[i][j] - (gdata[i][k] * gdata[k][j]);
			}
			gdata[i][k] = 0;
		}
		if (t_id == 0)
		{
			for (int i = 0; i < Num_thread - 1; i++)
				sem_wait(&sem_leader);//���̵߳ȴ������߳������ȥ
			for (int i = 0; i < Num_thread - 1; i++)
				sem_post(&sem_Elimination[i]);//�������������ȥ��֪ͨ������һ������
		}
		else {
			sem_post(&sem_leader);//�����߳�֪ͨleader�������ȥ
			sem_wait(&sem_Elimination[t_id - 1]);//�ȴ�������һ�ֵ�֪ͨ
		}
	}
	pthread_exit(NULL);
	return NULL;
}

//�̺߳���2��barrier�汾
void* threadFunc_barrier(void* Param)//����void���͵�ָ���ָ���������͵�����
{
	threadParam_t* p = (threadParam_t*)Param;//ǿ������ת�����߳����ݽṹ
	int t_id = p->t_id;//�̵߳ı�Ż�ȡ
	//��һ���̸߳���������������������Ҳ����к�����Ԫ����
	for (int k = 0; k < n; k++)
	{
		float32x4_t r0, r1, r2, r3;//��·���㣬�����ĸ�float�����Ĵ���
		if (t_id == 0)
		{
			r0 = vmovq_n_f32(gdata3[k][k]);//�ڴ治�������ʽ���ص������Ĵ�����
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				r1 = vld1q_f32(gdata3[k] + j);
				r1 = vdivq_f32(r1, r0);//������������λ���
				vst1q_f32(gdata3[k], r1);//���������·Ż��ڴ�
			}
			//��ʣ�಻��4�������ݽ�����Ԫ
			for (j; j < n; j++)
			{
				gdata3[k][j] = gdata3[k][j] / gdata3[k][k];
			}
			gdata3[k][k] = 1.0;
		}
		pthread_barrier_wait(&barrier_Division);//�ڽ��г���������ʹ��barrierֱ���߳�ͬ��

		//���񻮷֣�����Ϊ��λ���л���

		for (int i = k + 1 + t_id; i < n; i += Num_thread)
		{
			//�����������ѭ�����SIMD����
			r0 = vmovq_n_f32(gdata3[i][k]);
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				r1 = vld1q_f32(gdata3[k] + j);
				r2 = vld1q_f32(gdata3[i] + j);
				r3 = vmulq_f32(r0, r1);
				r2 = vsubq_f32(r2, r3);
				vst1q_f32(gdata3[i] + j, r2);
			}
			for (j; j < n; j++)
			{
				gdata3[i][j] = gdata3[i][j] - (gdata3[i][k] * gdata3[k][j]);
			}
			gdata3[i][k] = 0;
		}
		pthread_barrier_wait(&barrier_Elimination);//����Ԫ�������ٴν����߳�ͬ��
	}
	pthread_exit(NULL);
	return NULL;
}


//���ݳ�ʼ��
void Initialize(int N)
{
	for (int i = 0; i < N; i++)
	{
		//���Ƚ�ȫ��Ԫ����Ϊ0���Խ���Ԫ����Ϊ1
		for (int j = 0; j < N; j++)
		{
			gdata[i][j] = 0;
			gdata1[i][j] = 0;
			gdata2[i][j] = 0;
			gdata3[i][j] = 0;
		}
		gdata[i][i] = 1.0;
		//�������ǵ�λ�ó�ʼ��Ϊ�����
		for (int j = i + 1; j < N; j++)
		{
			gdata[i][j] = rand();
			gdata1[i][j] = gdata[i][j] = gdata2[i][j] = gdata3[i][j];
		}
	}
	for (int k = 0; k < N; k++)
	{
		for (int i = k + 1; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				gdata[i][j] += gdata[k][j];
				gdata1[i][j] += gdata1[k][j];
				gdata2[i][j] += gdata2[k][j];
				gdata3[i][j] += gdata3[k][j];
			}
		}
	}

}
//ƽ���㷨
void Normal_alg(int N)
{
	int i, j, k;
	for (k = 0; k < N; k++)
	{
		for (j = k + 1; j < N; j++)
		{
			gdata1[k][j] = gdata1[k][j] / gdata1[k][k];
		}
		gdata1[k][k] = 1.0;
		for (i = k + 1; i < N; i++)
		{
			for (j = k + 1; j < N; j++)
			{
				gdata1[i][j] = gdata1[i][j] - (gdata1[i][k] * gdata1[k][j]);
			}
			gdata1[i][k] = 0;
		}
	}
}

//��ȫ������SIMD�Ż�
void Par_alg_all(int n)
{
	int i, j, k;
	float32x4_t r0, r1, r2, r3;//��·���㣬�����ĸ�float�����Ĵ���
	for (k = 0; k < n; k++)
	{
		r0 = vmovq_n_f32(gdata2[k][k]);//�ڴ治�������ʽ���ص������Ĵ�����
		for (j = k + 1; j + 4 <= n; j += 4)
		{
			r1 = vld1q_f32(gdata2[k] + j);
			r1 = vdivq_f32(r1, r0);//������������λ���
			vst1q_f32(gdata2[k], r1);//���������·Ż��ڴ�
		}
		//��ʣ�಻��4�������ݽ�����Ԫ
		for (j; j < n; j++)
		{
			gdata2[k][j] = gdata2[k][j] / gdata2[k][k];
		}
		gdata2[k][k] = 1.0;
		//���϶�Ӧ������һ������ѭ���Ż���SIMD

		for (i = k + 1; i < n; i++)
		{

			r0 = vmovq_n_f32(gdata2[i][k]);
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				r1 = vld1q_f32(gdata2[k] + j);
				r2 = vld1q_f32(gdata2[i] + j);
				r3 = vmulq_f32(r0, r1);
				r2 = vsubq_f32(r2, r3);
				vst1q_f32(gdata2[i] + j, r2);
			}
			for (j; j < n; j++)
			{
				gdata2[i][j] = gdata2[i][j] - (gdata2[i][k] * gdata2[k][j]);
			}
			gdata2[i][k] = 0;
		}
	}
}


void pthread_NEON_semaphore()
{
	//��ʼ���ź���
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < Num_thread - 1; i++)
	{
		sem_init(&sem_Division[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	pthread_t* handles = new pthread_t[Num_thread];//�����߳̾��
	threadParam_t* param = new threadParam_t[Num_thread];//��Ҫ���ݵĲ������
	for (int t_id = 0; t_id < Num_thread; t_id++)
	{
		param[t_id].t_id = t_id;//���̲߳������ݣ��߳�����
		pthread_create(&handles[t_id], NULL, threadFunc_NEON, &param[t_id]);//�����̺߳���
	}
	for (int t_id = 0; t_id < Num_thread; t_id++)
		pthread_join(handles[t_id], NULL);
	sem_destroy(sem_Division);
	sem_destroy(sem_Elimination);
	sem_destroy(&sem_leader);
}

void pthread_NEON_barrier()
{
	//��ʼ��barrier
	pthread_barrier_init(&barrier_Division, NULL, Num_thread);
	pthread_barrier_init(&barrier_Elimination, NULL, Num_thread);
	pthread_t* handles = new pthread_t[Num_thread];//�����߳̾��
	threadParam_t* param = new threadParam_t[Num_thread];//��Ҫ���ݵĲ������
	for (int t_id = 0; t_id < Num_thread; t_id++)
	{
		param[t_id].t_id = t_id;//���̲߳������ݣ��߳�����
		pthread_create(&handles[t_id], NULL, threadFunc_barrier, &param[t_id]);//�����̺߳���
	}
	for (int t_id = 0; t_id < Num_thread; t_id++)
		pthread_join(handles[t_id], NULL);
	pthread_barrier_destroy(&barrier_Division);
	pthread_barrier_destroy(&barrier_Elimination);
}


int main()
{
	struct timeval begin, end;
	n = 1000;
	long long res;
	gettimeofday(&begin, NULL);
	Initialize(n);
	gettimeofday(&end, NULL);
	res = (1000 * 1000 * end.tv_sec + end.tv_usec) - (1000 * 1000 * begin.tv_sec + begin.tv_usec);
	cout << "initalize time:" << res << " us" << endl;

	gettimeofday(&begin, NULL);
	Normal_alg(n);
	gettimeofday(&end, NULL);
	res = (1000 * 1000 * end.tv_sec + end.tv_usec) - (1000 * 1000 * begin.tv_sec + begin.tv_usec);
	cout << "Normal time:" << res << " us" << endl;

	gettimeofday(&begin, NULL);
	pthread_NEON_semaphore();
	gettimeofday(&end, NULL);
	res = (1000 * 1000 * end.tv_sec + end.tv_usec) - (1000 * 1000 * begin.tv_sec + begin.tv_usec);
	cout << "pthread_NEON_semaphore time:" << res << " us" << endl;

	gettimeofday(&begin, NULL);
	pthread_NEON_barrier();
;	gettimeofday(&end, NULL);
	res = (1000 * 1000 * end.tv_sec + end.tv_usec) - (1000 * 1000 * begin.tv_sec + begin.tv_usec);
	cout << "pthread_NEON_barrier time:" << res << " us" << endl;

	gettimeofday(&begin, NULL);
	Par_alg_all(n);
	gettimeofday(&end, NULL);
	res = (1000 * 1000 * end.tv_sec + end.tv_usec) - (1000 * 1000 * begin.tv_sec + begin.tv_usec);
	cout << "SIMD time:" << res << " us" << endl;
}
