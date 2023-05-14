#include<iostream>
#include<pthread.h>
#include<stdio.h>
#include<arm_neon.h>
#include<sys/time.h>
#include<semaphore.h>//信号量包含的头文件
using namespace std;
alignas(16) float gdata[10000][10000];//进行对齐操作
float gdata2[10000][10000];
float gdata1[10000][10000];
float gdata3[10000][10000];
int n;
//定义线程数据结构
typedef struct {
	int t_id;	//线程编号
}threadParam_t;
const int Num_thread = 8;//8核CPU

sem_t sem_leader;		//进行除法运算的线程的信号量
sem_t sem_Division[Num_thread - 1];//判断除法是否完成的信号量
sem_t sem_Elimination[Num_thread - 1];//是否可以进行下一轮消去的信号量
//使用barrier版本
pthread_barrier_t barrier_Division;
pthread_barrier_t barrier_Elimination;


//线程函数1，semaphore版本
void* threadFunc_NEON(void* Param)//利用void类型的指针可指向任意类型的数据
{
	threadParam_t* p = (threadParam_t*)Param;//强制类型转换成线程数据结构
	int t_id = p->t_id;//线程的编号获取
	//让一个线程负责除法操作，其运算完后也会进行后续消元操作
	for (int k = 0; k < n; k++)
	{
		float32x4_t r0, r1, r2, r3;//四路运算，定义四个float向量寄存器
		if (t_id == 0)
		{
			r0 = vmovq_n_f32(gdata[k][k]);//初始化4个包含gdatakk的向量
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				r1 = vld1q_f32(gdata[k] + j);
				r1 = vdivq_f32(r1, r0);//将两个向量对位相除
				vst1q_f32(gdata[k], r1);//相除结果重新放回内存
			}
			//对剩余不足4个的数据进行消元
			for (j; j < n; j++)
			{
				gdata[k][j] = gdata[k][j] / gdata[k][k];
			}
			gdata[k][k] = 1.0;
		}
		else {
			sem_wait(&sem_Division[t_id - 1]);//阻塞直到除法操作完成
		}
		if (t_id == 0)//进行完除法操作唤醒其他线程
		{
			for (int i = 0; i < Num_thread - 1; i++)
			{
				sem_post(&sem_Division[i]);
			}
		}
		//任务划分，以行为单位进行划分

		for (int i = k + 1 + t_id; i < n; i += Num_thread)
		{
			//划分完对三重循环结合SIMD运算
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
				sem_wait(&sem_leader);//主线程等待其他线程完成消去
			for (int i = 0; i < Num_thread - 1; i++)
				sem_post(&sem_Elimination[i]);//若其他完成了消去，通知进行下一轮运算
		}
		else {
			sem_post(&sem_leader);//其余线程通知leader完成了消去
			sem_wait(&sem_Elimination[t_id - 1]);//等待进行下一轮的通知
		}
	}
	pthread_exit(NULL);
	return NULL;
}

//线程函数2，barrier版本
void* threadFunc_barrier(void* Param)//利用void类型的指针可指向任意类型的数据
{
	threadParam_t* p = (threadParam_t*)Param;//强制类型转换成线程数据结构
	int t_id = p->t_id;//线程的编号获取
	//让一个线程负责除法操作，其运算完后也会进行后续消元操作
	for (int k = 0; k < n; k++)
	{
		float32x4_t r0, r1, r2, r3;//四路运算，定义四个float向量寄存器
		if (t_id == 0)
		{
			r0 = vmovq_n_f32(gdata3[k][k]);//内存不对齐的形式加载到向量寄存器中
			int j;
			for (j = k + 1; j + 4 <= n; j += 4)
			{
				r1 = vld1q_f32(gdata3[k] + j);
				r1 = vdivq_f32(r1, r0);//将两个向量对位相除
				vst1q_f32(gdata3[k], r1);//相除结果重新放回内存
			}
			//对剩余不足4个的数据进行消元
			for (j; j < n; j++)
			{
				gdata3[k][j] = gdata3[k][j] / gdata3[k][k];
			}
			gdata3[k][k] = 1.0;
		}
		pthread_barrier_wait(&barrier_Division);//在进行除法操作后使用barrier直接线程同步

		//任务划分，以行为单位进行划分

		for (int i = k + 1 + t_id; i < n; i += Num_thread)
		{
			//划分完对三重循环结合SIMD运算
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
		pthread_barrier_wait(&barrier_Elimination);//在消元操作后再次进行线程同步
	}
	pthread_exit(NULL);
	return NULL;
}


//数据初始化
void Initialize(int N)
{
	for (int i = 0; i < N; i++)
	{
		//首先将全部元素置为0，对角线元素置为1
		for (int j = 0; j < N; j++)
		{
			gdata[i][j] = 0;
			gdata1[i][j] = 0;
			gdata2[i][j] = 0;
			gdata3[i][j] = 0;
		}
		gdata[i][i] = 1.0;
		//将上三角的位置初始化为随机数
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
//平凡算法
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

//对全部进行SIMD优化
void Par_alg_all(int n)
{
	int i, j, k;
	float32x4_t r0, r1, r2, r3;//四路运算，定义四个float向量寄存器
	for (k = 0; k < n; k++)
	{
		r0 = vmovq_n_f32(gdata2[k][k]);//内存不对齐的形式加载到向量寄存器中
		for (j = k + 1; j + 4 <= n; j += 4)
		{
			r1 = vld1q_f32(gdata2[k] + j);
			r1 = vdivq_f32(r1, r0);//将两个向量对位相除
			vst1q_f32(gdata2[k], r1);//相除结果重新放回内存
		}
		//对剩余不足4个的数据进行消元
		for (j; j < n; j++)
		{
			gdata2[k][j] = gdata2[k][j] / gdata2[k][k];
		}
		gdata2[k][k] = 1.0;
		//以上对应上述第一个二重循环优化的SIMD

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
	//初始化信号量
	sem_init(&sem_leader, 0, 0);
	for (int i = 0; i < Num_thread - 1; i++)
	{
		sem_init(&sem_Division[i], 0, 0);
		sem_init(&sem_Elimination[i], 0, 0);
	}
	pthread_t* handles = new pthread_t[Num_thread];//创建线程句柄
	threadParam_t* param = new threadParam_t[Num_thread];//需要传递的参数打包
	for (int t_id = 0; t_id < Num_thread; t_id++)
	{
		param[t_id].t_id = t_id;//将线程参数传递（线程名）
		pthread_create(&handles[t_id], NULL, threadFunc_NEON, &param[t_id]);//创建线程函数
	}
	for (int t_id = 0; t_id < Num_thread; t_id++)
		pthread_join(handles[t_id], NULL);
	sem_destroy(sem_Division);
	sem_destroy(sem_Elimination);
	sem_destroy(&sem_leader);
}

void pthread_NEON_barrier()
{
	//初始化barrier
	pthread_barrier_init(&barrier_Division, NULL, Num_thread);
	pthread_barrier_init(&barrier_Elimination, NULL, Num_thread);
	pthread_t* handles = new pthread_t[Num_thread];//创建线程句柄
	threadParam_t* param = new threadParam_t[Num_thread];//需要传递的参数打包
	for (int t_id = 0; t_id < Num_thread; t_id++)
	{
		param[t_id].t_id = t_id;//将线程参数传递（线程名）
		pthread_create(&handles[t_id], NULL, threadFunc_barrier, &param[t_id]);//创建线程函数
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
