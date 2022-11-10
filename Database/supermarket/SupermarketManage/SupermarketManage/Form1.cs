using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Data.SqlClient;
//表示在代码中引入微软发布的sqlserver数据库的ado.net程序集，
//引入后，就可以使用SqlConnection、SqlCommand等数据库对象来访问sqlserver数据库。

namespace SupermarketManage
{

    public partial class login : Form
    {
        public static login plogin=null ;
        public login()
        {
            plogin = this;
            InitializeComponent();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            //Data source=服务器名，Initial catalog=数据库名，User Id=sqlserver连接名，  
            //Password=数据库连接密码,integrated Security=True  
            string str = @"Data Source=LAPTOP-S7DJ8SCT;Initial catalog=SuperMarketCommodity;integrated Security=True";
            SqlConnection conn = new SqlConnection(str);//定义sql连接对象
            conn.Open();

            try { 
                 string selectsql = "Select * from login where account = '" + textBox1.Text + "' and password='" + textBox2.Text + "'";
                 SqlCommand cmd = new SqlCommand(selectsql, conn);//SqlCommand对象允许你指定在数据库上执行的操作的类型。  
                 cmd.CommandType = CommandType.Text;  
                 SqlDataReader sdr;
                 sdr = cmd.ExecuteReader();
                 if (sdr.Read())
                 {
                 
                    MainForm mainform = new MainForm();//登陆成功显示主界面  
                    mainform.Show();
                    this.Hide();

                  }
                 else
                 {
                     MessageBox.Show("登录失败! 请检查用户名或者密码重新登录！" );
                     return;
                 }
            }
            catch 
            {
                MessageBox.Show("登录失败! 请检查用户名或者密码重新登录！" );                  
                return;
            }
            conn.Close();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            string str = @"Data Source=LAPTOP-S7DJ8SCT;Initial catalog=SuperMarketCommodity;integrated Security=True";
            SqlConnection conn1 = new SqlConnection(str);//定义sql连接对象
            conn1.Open();

            try
            {
                string registersql = "insert into login values('" + textBox1.Text + "','" + textBox2.Text + "') ";
                SqlCommand cmd = new SqlCommand(registersql , conn1);  
                cmd.CommandType = CommandType.Text;
               // SqlDataReader sdr;
                //sdr = cmd.ExecuteReader();
                if (cmd.ExecuteNonQuery()>0)
                {
                    textBox1.Text="";
                    textBox2.Text = "";
                    MessageBox.Show("注册成功! 现在可以输入用户名和密码登录了！");
                    return;

                }
                else
                {
                    textBox1.Text = "";
                    textBox2.Text = "";
                    MessageBox.Show("注册失败! 该用户已存在或你的输入格式不正确，请修改用户名或者密码重新注册！");
                    return;
                }
            }
            catch
            {
                textBox1.Text = "";
                textBox2.Text = "";
                MessageBox.Show("注册失败! 请重新尝试！");
                return;
            }
            conn1.Close();
        }

        private void button3_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }

        private void label2_Click(object sender, EventArgs e)
        {

        }
    }
}
