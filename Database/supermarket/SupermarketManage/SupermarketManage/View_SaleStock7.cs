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

namespace SupermarketManage
{
    public partial class View_SaleStock : Form
    {
        public View_SaleStock()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            MainForm.pMainForm.Show();
            this.Close();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            string str = @"Data Source=LAPTOP-S7DJ8SCT;Initial catalog=SuperMarketCommodity;integrated Security=True";
            SqlConnection conn = new SqlConnection(str);

            
            try
            {
                conn.Open();
                //String newView = "use SuperMarketCommodity go if OBJECT_ID('tmp_sale','U')is not null drop table tmp_sale if OBJECT_ID('tmp_book','U')is not null drop table tmp_book if OBJECT_ID('sale_book','U')is not null drop table sale_book go if OBJECT_ID('stock_book_sale','V')is not null drop view stock_book_sale go create table tmp_book (Gi_ID int,tatal_book int) insert into tmp_book select 商品编号,sum(数量)as 进货量 from Invoice_book group by 商品编号 create table tmp_sale (Gs_ID int,tatal_sale int) insert into tmp_sale select 销售商品,sum(数量)as 进货量 from Sale_info group by 销售商品 create table sale_book (Gsi_ID int,tatal_book int,tatal_sale int) insert into sale_book select Gi_ID,tatal_book ,tatal_sale from tmp_book  left outer join tmp_sale on (Gi_ID=Gs_ID) go create view stock_book_sale as  select Goods_info.商品编号,Goods_info.商品名称,Goods_info.库存量,sale_book.tatal_book,sale_book.tatal_sale from Goods_info left outer join sale_book on (Goods_info.商品编号=sale_book.Gsi_ID) go select * from stock_book_sale";
                //String newView = "use SuperMarketCommodity go if OBJECT_ID('stock_book_sale','V')is not null drop view stock_book_sale go create view stock_book_sale as select Goods_info.商品编号,Goods_info.商品名称,Goods_info.库存量,sale_book.tatal_book,sale_book.tatal_sale from Goods_info left outer join sale_book on (Goods_info.商品编号=sale_book.Gsi_ID)";
                //SqlCommand cmd = new SqlCommand(newView, conn);//SqlCommand对象允许你指定在数据库上执行的操作的类型。  
                //cmd.CommandType = CommandType.Text;
               // cmd.ExecuteReader();
               // cmd.ExecuteNonQuery();

                SqlDataAdapter sqlDap = new SqlDataAdapter("Select * from stock_book_sale", conn);
                DataSet dds = new DataSet();
                sqlDap.Fill(dds);
                DataTable _table = dds.Tables[0];
                int count = _table.Rows.Count;
                dataGridView1.DataSource = _table;
            }
            catch(Exception ex)
            {
               // MessageBox.Show(ex.StackTrace);
                MessageBox.Show("抱歉 操作失败！ 请重试或检查是否连接错误");
                return;
            }
            finally {
                conn.Close();   
            }
            
        }
    }
}
