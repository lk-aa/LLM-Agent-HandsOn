# 1. 安装并启动Mysql服务

&emsp;&emsp;`DS-Agent` 项目涉及多个模块的管理和交互，其中包含`代理信息`、`会话线程`、`消息记录`、`知识库文件` 等数据的存储与检索。为了高效地组织和管理这些信息，数据库是必不可少的。具体来说，项目中的数据关系复杂，涉及多张表之间的 一对多、多对多 关系，比如代理与会话线程、会话线程与消息记录等。因此，在项目启动前，需要我们在本地安装并启动`Mysql`服务。

&emsp;&emsp;安装完成并启动`Mysql`服务后，我们可以使用一些可视化的工具来进行直观的测试连接。常用的像 👉 [workbench](https://www.mysql.com/products/workbench/)、 [DBeaver ](https://dbeaver.io/)、[Navicat](https://www.navicat.com/en/)等，大家按个人喜好选择就行。我们这里通过`Navicat`进行演示。

&emsp;&emsp; 首先启动 `Navicat` 客户端，进入主界面，创建新的连接： 在 `Navicat` 主界面左上方，点击 “连接” 按钮，选择 `MySQL`：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411281907607.png" width=100%></div>

&emsp;&emsp;在弹出的连接设置窗口中，填写你的 `MySQL` 数据库连接信息，其中：

- 连接名称：给你的连接起个名字，方便识别（例如：MateGen_Pro）。
- 主机名/IP 地址：如果安装在本地计算机上，可以填写 localhost 或 127.0.0.1。
- 端口：默认情况下，MySQL 使用端口 3306，除非你有修改，保持默认即可。
- 用户名：输入你的 MySQL 用户名（例如：root，或者你设置的其他用户名）。
- 密码：输入对应的密码（如果是 root 用户，默认是你安装时设置的密码）。

&emsp;&emsp;填写完连接信息后，点击窗口下方的 “测试连接” 按钮。如果连接成功，`Navicat` 会显示 "连接成功" 的提示。如果连接失败，则需要上述配置是否填写正确。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411281907608.png" width=100%></div>

&emsp;&emsp;最后，如果测试连接成功，点击 “确定” 按钮，`Navicat` 会保存连接并尝试连接到你的 `MySQL` 数据库，并显示在左侧的数据库列表，至此就可以选择并管理该数据库了。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411281907609.png" width=100%></div>

# 2. DS-Agent 后端API整体设计

&emsp;&emsp;**在`DS-Agent`项目中，将所有的后端服务通过`FastAPI` 封装成 `RestFul API`，与前端的`Vue3` 建立数据通信**。`FastAPI` 用于基于标准 `Python` 类型提示使用 `Python` 构建 `API`，使用 `ASGI` 的标准来构建 `Python Web` 框架和服务器。所以可以**简单理解为：`FastAPI` 是一个 `ASGI Web` 框架。通过它，可以让用户在前端页面上的产生的行为实时的与后端的服务建立通信。**

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202412061032397.png" width=100%></div>

# 3. 掌握SqlAlchemy的使用

&emsp;&emsp;通过`Python`代码环境与 `MySQL` 数据库做交互，有几种常用的方式，包括 `MySQL Connector/Python`、`PyMySQL`、`mysqlclient` 和 `SQLAlchemy` 等。每种方式都有其适用场景和优缺点。比如 **MySQL Connector/Python** 和 **PyMySQL** 是轻量级的 `MySQL` 驱动，适合直接执行 `SQL` 语句，提供基本的数据库连接和操作功能。非常适合需要简洁、快速连接数据库的场景。而**mysqlclient** 是 `MySQLdb` 的一个分支，提供了与 `MySQL` 的接口，支持多种特性，适用于性能要求较高的场景。尤其在复杂查询或大数据量操作时表现更佳。

&emsp;&emsp;`DS-Agent`项目中，**我们选择使用的是 SQLAlchemy**，其主要优点在于：它是一个非常易用的工具。**它提供了非常简单的方式使其可以通过 `Python` 类来操作数据库，避免了直接编写复杂的 `SQL` 语句**。我们只需要定义类和对象，`SQLAlchemy` 会自动处理数据库的增删查改操作，非常适合快速开发和管理数据库。总的来说，**它的易用性在于，你不需要深入了解 `SQL`，就能实现数据库交互，减少了开发的复杂性。**虽然易用，但 `SQLAlchemy` 在**构建企业级复杂系统时也有非常大的优势，主要体现在：灵活性高、事务管理强大、支持跨数据库迁移、自动化功能丰富、性能优化选项多**，同时它也能够提高代码的可维护性和可读性。这些特性使得它非常适合处理复杂的业务逻辑和大规模的系统开发。

&emsp;&emsp; `SQLAlchemy` 官方中有非常详细的使用说明和应用示例：https://www.sqlalchemy.org/

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411291123253.png" width=100%></div>

&emsp;&emsp;`SQLAlchemy`是一个 `Python` 的 ORM（对象关系映射）库，主要有两大组件：

- SQLAlchemy Core：低层次的 SQL 操作，提供原生的 SQL 语句构建。
- SQLAlchemy ORM：提供面向对象的操作方法，可以将 Python 对象映射到数据库表中。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411291509619.png" width=100%></div>

- **SQLAlchemy Core**

&emsp;&emsp;`SQLAlchemy Core` 提供了一个程序化的 `API` 来构建 `SQL` 查询，使得开发者可以通过 `Python` 的方法和函数来构造 `SQL` 语句。它通过多个核心组件共同构成了 `SQLAlchemy` 的基础功能，从而支持直接进行数据库交互如查询、删除等操作。如下图所示：

> SQLAlchemy Core Documents：https://docs.sqlalchemy.org/en/20/core/index.html

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411291149865.png" width=100%></div>

&emsp;&emsp;**`Schema` 指的是数据库结构的定义，包括表、列、索引等**，包括表（Table）：定义数据库表的结构，包含列、主键、外键等信息，列（Column）：定义表中的字段，指定字段类型、约束等。**而`Types` 是指数据库字段的类型**，`SQLAlchemy` 提供了一些常见的数据类型（例如 `Integer, String, Date, Float` 等），用于指定表列的类型。

&emsp;&emsp;使用`SQLAlchemy` 第一步需要在当前的`Python`运行环境中安装`SQLAlchemy`包，执行如下代码：


```python
# ! pip install sqlalchemy pymysql  ## sqlalchemy==2.0.36 pymysql==1.1.1
```

&emsp;&emsp;比如这里我们通过 `Table` 类定义了一个数据库表 `users`，这个表包含三个字段（列）：`id`、`name` 和 `age` ，每个字段（列）由 `Column` 类定义，其中包括字段名、数据类型（如 `Integer`、`String`）以及其他约束（如主键）。如下所示：


```python
from sqlalchemy import Table, Column, Integer, String, MetaData

# 创建 MetaData 实例，它将存储表和列的元数据
metadata = MetaData()

# 定义一个数据库表 'users'，这个表将包含 id, name, 和 age 三个字段
users = Table('users', metadata,
              Column('id', Integer, primary_key=True),  # 'id' 是主键，类型为整数
              Column('name', String(50)),               # 'name' 字段是长度为50的字符串
              Column('age', Integer))                   # 'age' 字段是整数类型
```

&emsp;&emsp;`Engine` 是与数据库的桥梁，负责管理与数据库的连接，执行 `SQL` 查询以及管理数据库事务。其执行逻辑如下图所示：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411291256313.png" width=100%></div>

&emsp;&emsp;`DBAPI` 是`Python Database API`的简写。在`Python` 中用于与数据库交互的标准接口规范。它定义了 `Python` 程序与关系型数据库之间交互的标准化方法，使开发者可以通过统一的方式访问不同类型的数据库（如 MySQL、PostgreSQL、SQLite 等）。常见的 `DB-API` 示例代码如下所示：


```python
import pymysql

# 步骤 1: 创建数据库
connection = pymysql.connect(host='localhost', user='root', password='lkaa')    # 注意：这里替换成自己的数据库配置信息
try:
    with connection.cursor() as cursor:
        cursor.execute('CREATE DATABASE IF NOT EXISTS mydb')
        print("Database 'mydb' created or already exists.")
finally:
    connection.close()

# 步骤 2: 连接到 'mydb' 并创建表
connection = pymysql.connect(host='localhost', user='root', password='lkaa', database='mydb')  # 注意：这里替换成自己的数据库配置信息
try: 
    with connection.cursor() as cursor:
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(50),
            age INT
        )
        '''
        cursor.execute(create_table_query)
        print("Table 'users' created or already exists.")
finally:
    connection.close()

# 步骤 3: 插入数据
connection = pymysql.connect(host='localhost', user='root', password='lkaa', database='mydb')  # 注意：这里替换成自己的数据库配置信息
try:
    with connection.cursor() as cursor:
        insert_query = "INSERT INTO users (name, age) VALUES (%s, %s)"
        users_data = [
            ('Zs', 28),
            ('Ls', 34),
        ]
        cursor.executemany(insert_query, users_data)
        connection.commit()
        print("Data inserted into 'users' table.")
finally:
    connection.close()

# 步骤 4: 查询数据
connection = pymysql.connect(host='localhost', user='root', password='lkaa', database='mydb')   # 注意：这里替换成自己的数据库配置信息
try:
    with connection.cursor() as cursor:
        query = "SELECT id, name, age FROM users"
        cursor.execute(query)
        result = cursor.fetchall()
        for row in result:
            print(row)
finally:
    connection.close()
```

    Database 'mydb' created or already exists.
    Table 'users' created or already exists.
    Data inserted into 'users' table.
    (1, 'Zs', 28)
    (2, 'Ls', 34)
    

&emsp;&emsp;此时可以通过`Navicat`查看数据库中是否依次执行了创建表，插入数据的操作：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411291309294.png" width=100%></div>

&emsp;&emsp;`SQLAlchemy` 本身并不直接与数据库交互，而是通过各种数据库驱动（即 `DB-API` 规范的实现）来实现与数据库的连接和交互。换句话说，`SQLAlchemy` 作为一个中间层，提供了一个通用的接口来支持不同数据库的操作，而实际的数据库通信是通过 `DB-API` 实现的。它使用统一的 `URL` 连接字符串语法来表示如何连接到不同的数据库。这个 `URL` 包括了数据库类型、数据库驱动（即 DB-API）和连接所需的相关参数。比如我们要使用的`Mysql`服务，建立连接的方法如下代码所示：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411291256313.png" width=100%></div>


```python
import pymysql
from sqlalchemy import create_engine, text

# 创建一个连接到 MySQL 服务器的引擎，注意我们暂时不连接到某个具体的数据库
engine = create_engine('mysql+pymysql://root:lkaa@localhost', echo=True)  # 注意： 这里替换成自己本地实际的Mysql连接信息
# echo 表示展示连接的过程, 会输出很多信息

# 使用该引擎创建一个连接，并执行创建数据库的 SQL 语句
with engine.connect() as connection:

    # text() 是 SQLAlchemy 提供的一个方法，用于包装原生的 SQL 语句，并使其成为可执行的对象
    connection.execute(text('CREATE DATABASE IF NOT EXISTS my_sqlalchemy_db'))  # 创建一个名为 'my_sqlalchemy_db' 的数据库
    print("Database 'my_sqlalchemy_db' created or already exists.")
```

    2025-03-04 21:40:58,465 INFO sqlalchemy.engine.Engine SELECT DATABASE()
    2025-03-04 21:40:58,466 INFO sqlalchemy.engine.Engine [raw sql] {}
    2025-03-04 21:40:58,467 INFO sqlalchemy.engine.Engine SELECT @@sql_mode
    2025-03-04 21:40:58,468 INFO sqlalchemy.engine.Engine [raw sql] {}
    2025-03-04 21:40:58,469 INFO sqlalchemy.engine.Engine SELECT @@lower_case_table_names
    2025-03-04 21:40:58,469 INFO sqlalchemy.engine.Engine [raw sql] {}
    2025-03-04 21:40:58,470 INFO sqlalchemy.engine.Engine BEGIN (implicit)
    2025-03-04 21:40:58,471 INFO sqlalchemy.engine.Engine CREATE DATABASE IF NOT EXISTS my_sqlalchemy_db
    2025-03-04 21:40:58,471 INFO sqlalchemy.engine.Engine [generated in 0.00118s] {}
    Database 'my_sqlalchemy_db' created or already exists.
    2025-03-04 21:40:58,476 INFO sqlalchemy.engine.Engine ROLLBACK
    

&emsp;&emsp;在 `SQLAlchemy` 中，通常是直接在连接的数据库上进行操作，若需要创建数据库，通常先用一个数据库连接来执行创建数据库的 `SQL` 语句。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411291322389.png" width=100%></div>

&emsp;&emsp;在这段代码中的 `engine = create_engine('mysql+pymysql://root:snowball2019@localhost', echo=True)` 就是 `Dialect`。  它是 `SQLAlchemy` 用来支持多种数据库的组件，处理不同数据库之间的语法差异。每种数据库都需要一个特定的 `Dialect`，比如如果连接的不是 `Mysql`而是 `postgresql`，则代码就应该修改成：

```python
# 创建一个连接 PostgreSQL 数据库的 Engine
engine = create_engine('postgresql://username:password@localhost/mydb')
```

> SqlAlchemy 支持的 Dialects：https://docs.sqlalchemy.org/en/20/dialects/index.html

&emsp;&emsp;另外一个`Connecting Pool(连接池)` 则是 `SQLAlchemy` 中的一个重要优化功能，它可以允许复用数据库连接，从而减少连接的创建和销毁带来的开销。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411291256313.png" width=100%></div>


```python
from sqlalchemy import create_engine

# 创建一个带有连接池的 Engine，设置池大小为 10，最大溢出连接数为 20
engine = create_engine(
    'mysql+pymysql://root:lkaa@localhost/my_sqlalchemy_db',  # 这里添加具体的数据库名称
    pool_size=10,  # 设置连接池大小为10
    max_overflow=20  # 最大溢出连接数为20
)
```

> SqlAlchemy Collection Pooling Configuration：https://docs.sqlalchemy.org/en/20/core/engines.html#sqlalchemy.create_engine

&emsp;&emsp;建立数据量连接后，相较于`DB-API`通过手动编写原生 `SQL` 语句来创建数据库表，`SQLAlchemy`则可以通过高层的 `Python API` 来定义表结构，然后由 `SQLAlchemy` 自动生成相应的 `SQL` 语句并执行，使用方法如下：

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411291149865.png" width=100%></div>


```python
from sqlalchemy import Table, Column, Integer, String, MetaData

# 创建 MetaData 实例，它将存储表和列的元数据
metadata = MetaData()

# 定义一个数据库表 'users'，这个表将包含 id, name, 和 age 三个字段
users = Table('users', metadata,
              Column('id', Integer, primary_key=True),  # 'id' 是主键，类型为整数
              Column('name', String(50)),               # 'name' 字段是长度为50的字符串
              Column('age', Integer))                   # 'age' 字段是整数类型
```


```python
with engine.connect() as connection:
    # 通过 SQLAlchemy 的 `metadata.create_all()` 方法自动生成并执行 `CREATE TABLE` 语句
    metadata.create_all(connection)
    print("Table 'users' created or already exists.")
```

    Table 'users' created or already exists.
    

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411291406912.png" width=100%></div>

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411291149865.png" width=100%></div>

&emsp;&emsp;最后看 `SQL Expression Language`。它是 `SQLAlchemy` 提供的高级接口，用于通过 `Python` 代码构建 `SQL` 查询。提供了比 `DB-API` 更高的抽象层，允许我们通过 `Python` 表达式创建查询，而不需要手动编写 `SQL` 语句。它会将 `Python` 表达式转换成对应的 `SQL` 语句并执行，从而使数据库操作更加简洁、直观。核心组件如下：
- Table：表示数据库中的一张表。
- Column：表示表中的一列。
- select()：构建 SELECT 查询。
- insert()：构建 INSERT 查询。
- update()：构建 UPDATE 查询。
- delete()：构建 DELETE 查询。

&emsp;&emsp;我们使用 `SQL Expression Language` 插入一条记录到 `users` 表中，代码如下所示：


```python
from sqlalchemy import insert

# 插入一条新记录
with engine.connect() as connection:
    stmt = insert(users).values(name='Muyu', age=28)  # 插入数据
    connection.execute(stmt)  # 执行插入操作

    connection.commit()  # 提交事务
    print("New record inserted into 'users' table.")
```

    New record inserted into 'users' table.
    

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411291415550.png" width=100%></div>

&emsp;&emsp;接下来，我们使用 `SQL Expression Language` 来查询 `users` 表中的所有记录，代码如下：


```python
from sqlalchemy import select

# 连接到数据库
with engine.connect() as connection:
    # 使用 SQL Expression Language 创建 SELECT 查询
    stmt = select(users.columns)  # 查询所有列
    result = connection.execute(stmt)  # 执行查询
    
    # 打印查询结果
    for row in result:
        print(row)  # 输出每一行
```

    (1, 'Muyu', 28)
    

&emsp;&emsp;以上就是`SQLAlchemy Core`中各个组件的详细使用方法。它作为`SQLAlchemy`中的低级组件，提供了与数据库交互的基础设施，直接使用 `SQL` 表达式语言（SQL Expression Language）来构建、执行 `SQL` 查询。通过 `Core`，开发者可以直接操作数据库的表、列和其他数据库元素。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411291149865.png" width=100%></div>

&emsp;&emsp;`Core` 提供的工具方法并没有将数据库表和 `Python` 对象直接关联。也就是说，使用 `Core` 时，只能通过类似 `select()`、`insert()`、`update()` 等 `SQL` 表达式来执行 `SQL` 操作。

- **SQLAlchemy ORM**

&emsp;&emsp;`ORM`（对象关系映射）是 `SQLAlchemy` 中的一个高级功能，它可以使数据库中的表与 `Python` 类（对象）之间建立映射关系。`ORM` 让我们能够像操作普通 `Python` 对象一样操作数据库记录，即以面向对象的方式操作数据库。其流程如下所示：从 `Engine` 到`Base`,`Mapped Class`，再到 `Session` 和 `Query`，每个步骤都有其明确的作用，形成了 `SQLAlchemy ORM` 中的标准操作流程。其中 `Engine` 负责数据库连接，`Base` 定义了映射类的结构，`Mapped Class`代表数据库中的一张表，并将类的属性映射到数据库中的字段,通过继承自 `Base`，每个类就自动具备了 `ORM` 映射的能力。`Session` 管理数据库操作，`Query` 用来执行查询。

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411291510075.png" width=100%></div>

&emsp;&emsp;这张流程图非常清晰地解释了 `SQLAlchemy ORM` 的核心概念以及它们之间的关系。接下来，我们使用`SQLAlchemy ORM` 来完成一套从建立连接、创建表、插入数据、查询数据的完整流程。首先，通过`create_engine()`用来创建一个连接数据库的引擎:


```python
from sqlalchemy import create_engine

# 1. 创建引擎
engine = create_engine('mysql+pymysql://root:lkaa@localhost/my_sqlalchemy_db', echo=True)
```

&emsp;&emsp;通过`declarative_base()`为 `ORM` 定义一个基类。作用是为 `ORM` 映射类（比如下面即将定义的 `User` 类）提供一个基类，它允许我们使用 `SQLAlchemy` 的 `ORM` 功能进行映射和数据库操作。具体来说，它为我们定义的类提供了数据库表的元数据和映射功能，使得这些类可以与数据库中的表相对应


```python
from sqlalchemy.orm import declarative_base

# 2. 定义 ORM 映射类
Base = declarative_base()
```

&emsp;&emsp;创建映射类，`User` 类通过继承 `Base` 来映射数据库中的 `users` 表， 其中 `__tablename__` 属性定义了该类对应的数据库表名。`Column` 用来定义表的列，`Integer` 和 `String` 定义了列的类型，`primary_key=True` 表示该列是主键，代码如下所示：


```python
from sqlalchemy import Column, Integer, String

# 3. 定义映射类
class UsersORM(Base):
    __tablename__ = 'users_orm'  # 对应数据库中的表名
    
    id = Column(Integer, primary_key=True)  # 主键列
    name = Column(String(50))  # 名字字段
    age = Column(Integer)  # 年龄字段

    def __repr__(self):
        return f"<User(id={self.id}, name={self.name}, age={self.age})>"
```

&emsp;&emsp;`Base.metadata.create_all(engine)`会根据定义的映射类创建对应的数据库表。如果表已经存在，它不会做任何操作。


```python
# 4. 创建表（如果不存在）
Base.metadata.create_all(engine)  # 这会在数据库中创建 `users_orm` 表
```

    2025-03-04 22:04:52,229 INFO sqlalchemy.engine.Engine SELECT DATABASE()
    2025-03-04 22:04:52,230 INFO sqlalchemy.engine.Engine [raw sql] {}
    2025-03-04 22:04:52,231 INFO sqlalchemy.engine.Engine SELECT @@sql_mode
    2025-03-04 22:04:52,232 INFO sqlalchemy.engine.Engine [raw sql] {}
    2025-03-04 22:04:52,233 INFO sqlalchemy.engine.Engine SELECT @@lower_case_table_names
    2025-03-04 22:04:52,234 INFO sqlalchemy.engine.Engine [raw sql] {}
    2025-03-04 22:04:52,235 INFO sqlalchemy.engine.Engine BEGIN (implicit)
    2025-03-04 22:04:52,236 INFO sqlalchemy.engine.Engine DESCRIBE `my_sqlalchemy_db`.`users_orm`
    2025-03-04 22:04:52,236 INFO sqlalchemy.engine.Engine [raw sql] {}
    2025-03-04 22:04:52,240 INFO sqlalchemy.engine.Engine 
    CREATE TABLE users_orm (
    	id INTEGER NOT NULL AUTO_INCREMENT, 
    	name VARCHAR(50), 
    	age INTEGER, 
    	PRIMARY KEY (id)
    )
    
    
    2025-03-04 22:04:52,240 INFO sqlalchemy.engine.Engine [no key 0.00089s] {}
    2025-03-04 22:04:52,264 INFO sqlalchemy.engine.Engine COMMIT
    

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411291447255.png" width=100%></div>

&emsp;&emsp;接下来需要创建一个`session`，这是一个实例化的会话对象，所有的数据库操作都通过 `session` 来执行。创建的方法如下：


```python
from sqlalchemy.orm import sessionmaker

# Step 5. 创建 Session 会话
Session = sessionmaker(bind=engine)  # 返回一个 Session 类，通过 Session 可以对数据库进行增删改查等操作。
session = Session()   # 实例化一个会话对象，所有的数据库操作都通过 session 来执行。
```

&emsp;&emsp;最后，则可以通过 `session`对象进行增删改查等操作。


```python
new_user = UsersORM(name='Muyu', age=28)
session.add(new_user)  # 将数据添加到会话
session.commit()  # 提交事务
```

    2024-12-06 20:16:06,269 INFO sqlalchemy.engine.Engine BEGIN (implicit)
    2024-12-06 20:16:06,271 INFO sqlalchemy.engine.Engine INSERT INTO users_orm (name, age) VALUES (%(name)s, %(age)s)
    2024-12-06 20:16:06,273 INFO sqlalchemy.engine.Engine [generated in 0.00138s] {'name': 'Muyu', 'age': 28}
    2024-12-06 20:16:06,276 INFO sqlalchemy.engine.Engine COMMIT
    

<div align=center><img src="https://muyu20241105.oss-cn-beijing.aliyuncs.com/images/202411291451797.png" width=100%></div>


```python
users = session.query(UsersORM).all()  # 查询所有用户
print("Users in database:")
for user in users:
    print(user)
```

    2024-12-06 20:17:07,069 INFO sqlalchemy.engine.Engine BEGIN (implicit)
    2024-12-06 20:17:07,071 INFO sqlalchemy.engine.Engine SELECT users_orm.id AS users_orm_id, users_orm.name AS users_orm_name, users_orm.age AS users_orm_age 
    FROM users_orm
    2024-12-06 20:17:07,072 INFO sqlalchemy.engine.Engine [generated in 0.00145s] {}
    Users in database:
    <User(id=1, name=Muyu, age=28)>
    

&emsp;&emsp;通过上面的步骤，我们就已经使用 `SQLAlchemy ORM` 完成了从创建数据库连接、定义映射类、创建表、插入数据到查询数据的完整流程。`ORM` 让开发者能够以面向对象的方式操作数据库，减少了直接写 `SQL` 查询的复杂性。同时，`SQLAlchemy ORM` 也会自动处理对象与数据库表之间的映射，从而提高了开发效率。

> SQLAlchemy ORM：https://docs.sqlalchemy.org/en/20/orm/index.html

&emsp;&emsp;掌握到这里，基本的`SQLAlchemy ORM`的原理和使用就已经可以灵活应对了，这里我们给出`ORM`和`Core`的对比：

| 特性                   | SQLAlchemy ORM                                   | SQLAlchemy Core                                 |
|------------------------|--------------------------------------------------|-------------------------------------------------|
| **操作方式**           | 使用 Python 类和对象来操作数据                 | 使用 SQL 表和列对象来构建和执行 SQL 语句       |
| **抽象层次**           | 高层次：面向对象编程                           | 低层次：面向 SQL 编程                           |
| **使用场景**           | 适合需要关系映射和面向对象编程的应用            | 适合需要灵活控制 SQL、复杂查询的应用          |
| **关系管理**           | 自动处理一对多、多对多关系                      | 需要手动管理表之间的关系                       |
| **学习曲线**           | 相对较高，适合有 ORM 经验的开发者               | 较低，适合熟悉 SQL 和数据库操作的开发者        |
| **灵活性**             | 较低，因为是高层次的封装                       | 较高，可以直接操作 SQL，适应复杂查询需求      |

&emsp;&emsp;总的来说，如果你想通过 `Python` 类来操作数据库，自动映射表和记录，并且希望快速进行增、查、改、删操作，那么 **SQLAlchemy ORM** 是最合适的选择。如果你希望直接构建和执行 SQL 查询，或者需要更复杂的查询和性能优化，那么 **SQLAlchemy Core** 提供了更高的灵活性。大家可以根据需求选择适合的部分，甚至可以在同一个项目中结合使用这两者，ORM 用于高层次的操作，Core 用于更底层的 SQL 构建和执行。

&emsp;&emsp;这里采用的操作`Mysql`的方法就是`SQLAlchemy ORM`语法。
