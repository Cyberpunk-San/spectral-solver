# redis_test_fixed.py
import asyncio
import redis.asyncio as redis
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("redis_test")

async def test_redis():
    try:
        # Try to connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Test basic operations
        await r.ping()
        logger.info("✅ Redis ping successful!")
        
        # Test set/get
        await r.set('test_key', 'hello_redis')
        value = await r.get('test_key')
        logger.info(f"✅ Redis set/get test: {value}")
        
        # Test hash operations - use older syntax for compatibility
        try:
            # Try new syntax first
            await r.hset('test_hash', mapping={'field1': 'value1', 'field2': 'value2'})
            logger.info("✅ Redis hset with mapping syntax works")
        except Exception as e:
            logger.warning(f"hset mapping syntax failed, using old syntax: {e}")
            # Use old syntax
            await r.hset('test_hash', 'field1', 'value1')
            await r.hset('test_hash', 'field2', 'value2')
            logger.info("✅ Redis hset with old syntax works")
        
        hash_data = await r.hgetall('test_hash')
        logger.info(f"✅ Redis hash test: {hash_data}")
        
        # Test other operations we use
        await r.sadd('test_set', 'member1', 'member2')
        set_members = await r.smembers('test_set')
        logger.info(f"✅ Redis set test: {set_members}")
        
        await r.lpush('test_list', 'item1', 'item2')
        logger.info("✅ Redis list operations work")
        
        # Cleanup test data
        await r.delete('test_key', 'test_hash', 'test_set', 'test_list')
        logger.info("✅ Cleanup successful")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_redis())
    if result:
        print("🎉 Redis is working perfectly!")
    else:
        print("💥 Redis connection issues detected")